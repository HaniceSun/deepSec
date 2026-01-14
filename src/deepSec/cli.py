import argparse
from importlib import resources
from .downloader import DataDownloader
from .preprocess import DataProcessor
from .dataset import CustomDataset
from .dataset import CustomDatasetPredPaired
from .dataset import CustomDatasetPredSingle
from .trainer import Trainer


def get_parser():
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest='command', required=True)

    p1 = subparsers.add_parser("download-training-data", help="get training data from Ensembl")
    p1.add_argument('--output_dir', type=str, default='data', help='output directory to save the downloaded data')
    p1.add_argument('--from_ensembl', type=str, default='True', help='whether to download data from ftp.ensembl.org')
    p1.add_argument('--ensemble_version', type=str, default='111', help='Ensembl version to download data from')
    p1.add_argument('--from_ensemblgenomes', type=str, default='True', help='whether to download data from ftp.ensemblgenomes.org')
    p1.add_argument('--ensemblgenomes_version', type=str, default='62', help='EnsemblGenomes version to download data from')

    p2 = subparsers.add_parser("preprocess", help="preprocess downloaded training data")
    p2.add_argument('--input_dir', type=str, default='data', help='input directory of downloaded data')
    p2.add_argument('--output_file', type=str, default='Ensembl_Protein2Transcript.txt.gz', help='output file of the processed data')
    p2.add_argument('--flank', type=int, default=50, help='length of flanking sequences to extract')
    p2.add_argument('--down_sampling_fold', type=int, default=100, help='down-sampling fold for negative samples')

    p3 = subparsers.add_parser("torch-dataset", help="generate torch dataset from the processed files")
    p3.add_argument('--input_file', type=str, default='Ensembl_Protein2Transcript_Upos_FeatureSeq_DownSampling100.txt.gz', help='input file of the processed data')
    p3.add_argument('--split_by_species', type=str, default='true', help='split the dataset to train, val, and test datasets by species')
    p3.add_argument('--random_split_ratio', type=str, default='0.8,0.1,0.1', help='split the dataset to train, val, and test datasets using the ratios when split_by_species is false')

    p4 = subparsers.add_parser("train", help="train the deep learning models")
    p4.add_argument('--config_file', type=str, default='config.yaml', help='configuration file for model training')
    p4.add_argument('--model_name', type=str, default='SpliceAI', help='the model to train')
    p4.add_argument('--train_file', type=str, default='dataset_train.pt', help='training dataset file')
    p4.add_argument('--val_file', type=str, default='dataset_val.pt', help='validation dataset file')
    p4.add_argument('--metrics_file', type=str, default='metrics.txt', help='metrics output file')
    p4.add_argument('--lr_lambda', type=str, default=None, help='learning rate as a string seperated by comma for different epochs')
    p4.add_argument('--resume_epoch', type=int, default=None, help='resume training from a specific epoch if specified')

    p5 = subparsers.add_parser("test", help="test a trained model")
    p5.add_argument('--config_file', type=str, default='config.yaml', help='configuration file for model training')
    p5.add_argument('--model_name', type=str, default='SpliceAI', help='the model to test')
    p5.add_argument('--test_file', type=str, default='dataset_test.pt', help='test dataset file')
    p5.add_argument('--epoch', type=int, default=4, help='the epoch of the trained model to load')

    p6 = subparsers.add_parser("predict", help="predict using a trained model")
    p6.add_argument('--config_file', type=str, default='config.yaml', help='configuration file for model training')
    p6.add_argument('--model_name', type=str, default='SpliceAI', help='the model to test')
    p6.add_argument('--pred_seq', type=str, default='ATCG,ATCG', help='sequences to predict, separated by comma')
    p6.add_argument('--pred_file', type=str, default='dataset_pred.pt', help='pred dataset file')
    p6.add_argument('--pred_file_paired', type=str, default='false', help='if the sequences are paired (REF and ALT) in the pred_file')
    p6.add_argument('--epoch', type=int, default=4, help='the epoch of the trained model to load')
    p6.add_argument('--out_file', type=str, default='predicted.txt', help='prediction output file')

    p7 = subparsers.add_parser("saliency", help="generate saliency map using a trained model")
    p7.add_argument('--sal_seq', type=str, default='ATCG,ATCG', help='sequences to get saliency map, separated by comma')
    p7.add_argument('--out_file', type=str, default='saliency.txt', help='saliency output file')
    p7.add_argument('--model_name', type=str, default='SpliceAI', help='the model to test')
    p7.add_argument('--epoch', type=int, default=4, help='the epoch of the trained model to load')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    dd = DataDownloader()
    dp = DataProcessor()
    if args.command == 'download-training-data':
        from_ensembl = args.from_ensembl.lower() in ('true', '1', 'yes')
        from_ensemblgenomes = args.from_ensemblgenomes.lower() in ('true', '1', 'yes')
        dd.download_training_data(from_ensembl=from_ensembl, ensemble_version=args.ensemble_version,
                                  from_ensemblgenomes=from_ensemblgenomes, ensemblgenomes_version=args.ensemblgenomes_version,
                                  out_dir=args.output_dir)

    elif args.command == 'preprocess':
        print('mapping pep, cds, and cdna ...', flush=True)
        dp.pep2cds2cdna(in_dir=args.input_dir, out_file=args.output_file)
        print('getting U positions...', flush=True)
        dp.get_pos_of_U(in_file=args.output_file)
        print('getting feature sequences...', flush=True)
        dp.get_feature_seq(in_file=args.output_file, flank=args.flank)
        print('down sampling negative samples...', flush=True)
        dp.down_sampling_negtive_samples(in_file=args.output_file, n_fold=args.down_sampling_fold)
    elif args.command == 'torch-dataset':
        ds = CustomDataset(args.input_file)
        split_by_species = (args.split_by_species.lower() in ['true', 'yes', '1'])
        ratio = [float(x) for x in args.random_split_ratio.split(',')]
        if split_by_species:
            ds.split_save_dataset(split_by_species=True)
        else:
            ds.split_save_dataset(ratio=ratio, split_by_species=False)
    elif args.command == 'train':
        lr_lambda = None
        if args.lr_lambda:
            lr_lambda = [float(x) for x in args.lr_lambda.split(',')]

        trainer = Trainer(config_file=args.config_file, model_name=args.model_name,
                          train_file=args.train_file, val_file=args.val_file,
                          metrics_file=args.metrics_file, lr_lambda=lr_lambda)
        trainer.count_parameters()
        trainer.run(resume_epoch=args.resume_epoch)
    elif args.command == 'test':
        trainer = Trainer(config_file=args.config_file, model_name=args.model_name,
                          test_file=args.test_file)
        trainer.test(epoch=args.epoch)
    elif args.command == 'predict':
        trainer = Trainer(config_file=args.config_file, model_name=args.model_name)
        pred_seq = [s.strip() for s in args.pred_seq.split(',')]
        pred_file_paired = args.pred_file_paired.lower() in ('true', '1', 'yes')
        trainer.predict(epoch=args.epoch, pred_seq=pred_seq,
                        pred_file=args.pred_file, pred_file_paired=pred_file_paired, out_file=args.out_file)
    elif args.command == 'saliency':
        trainer = Trainer(config_file=args.config_file, model_name=args.model_name)
        sal_seq = [s.strip() for s in args.sal_seq.split(',')]
        trainer.saliency_map(epoch=args.epoch, sal_seq=sal_seq, out_file=args.out_file)


if __name__ == '__main__':
    main()
