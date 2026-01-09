#!/usr/bin/env python
import os
import numpy as np
import gzip
import glob
import subprocess

class DataProcessor():
    def __int__(self):
        pass

    def pep2cds2cdna(self, in_dir='data', out_file='Ensembl_Protein2Transcript.txt.gz'):
        ouFile = gzip.open(out_file, 'wt')
        ouFile.write('\t'.join(['species', 'gene_id', 'gene_name', 'transcript_id', 'protein_id', 'cdna', 'cds', 'pep', 'source']) + '\n')

        in_files = glob.glob(f'{in_dir}/**/**OneLine.fa.gz', recursive=True)
        S = {}
        for f in in_files:
            if f.find('abinitio') == -1:
                if f.find('.cdna.') != -1 or f.find('.cds.') != -1 or f.find('.pep.') != -1:
                    k = f.split('/')[-1].replace('.cdna.', '.').replace('.cds.', '.').replace('.pep.', '.')
                    S.setdefault(k, [])
                    S[k].append(f)

        for k in sorted(S):
            print(f'Processing {k} {len(S[k])}', flush=True)
            if len(S[k]) == 3:
                D = {}
                for in_file in sorted(S[k]):
                    inFile = gzip.open(in_file, 'rt')
                    while True:
                        line1 = inFile.readline().strip()
                        line2 = inFile.readline().strip()
                        if line1:
                            if in_file.find('.cdna.') != -1:
                                transcript_id = line1.split()[0][1:]
                                gene_id = line1.split('gene:')[-1].split()[0]
                                gene_name = '.'
                                if line1.find('gene_symbol:') != -1:
                                    gene_name = line1.split('gene_symbol:')[-1].split()[0]
                                sp = in_file.split('.cdna.')[0].split('/')[-1]
                                ## species, gene_id, gene_name, transcript_id, protein_id, cdna, cds, pep, source
                                D.setdefault(transcript_id, ['.', '.', '.', '.', '.', '.', '.', '.', '.'])
                                D[transcript_id][0] = sp
                                D[transcript_id][1] = gene_id
                                D[transcript_id][2] = gene_name
                                D[transcript_id][3] = transcript_id
                                D[transcript_id][5] = line2
                                D[transcript_id][8] = in_file.split('data/')[-1].split('/')[0]
                            elif in_file.find('.pep.') != -1:
                                protein_id = line1.split()[0][1:]
                                gene_id = line1.split('gene:')[-1].split()[0]
                                transcript_id = line1.split('transcript:')[-1].split()[0]
                                gene_name = '.'
                                if line1.find('gene_symbol:') != -1:
                                    gene_name = line1.split('gene_symbol:')[-1].split()[0]
                                D.setdefault(transcript_id, ['.', '.', '.', '.', '.', '.', '.', '.', '.'])
                                D[transcript_id][4] = protein_id
                                D[transcript_id][7] = line2
                            elif in_file.find('.cds.') != -1:
                                transcript_id = line1.split()[0][1:]
                                gene_id = line1.split('gene:')[-1].split()[0]
                                gene_name = '.'
                                if line1.find('gene_symbol:') != -1:
                                    gene_name = line1.split('gene_symbol:')[-1].split()[0]
                                D.setdefault(transcript_id, ['.', '.', '.', '.', '.', '.', '.', '.', '.'])
                                D[transcript_id][6] = line2
                        else:
                            break
                    inFile.close()
        
                for k in D:
                    if D[k][-3] != '.' and D[k][-2] != '.' and D[k][-1] != '.':
                        ouFile.write(('\t'.join(D[k]) + '\n'))
        ouFile.close()

    def get_pos_of_U(self, in_file='Ensembl_Protein2Transcript.txt.gz'):
        ouFile = gzip.open(in_file.replace('.txt.gz', '_Upos.txt.gz'), 'wt')
        inFile = gzip.open(in_file, 'rt')
        head = inFile.readline().strip()
        ouFile.write(head + '\t' + '\t'.join(['cdna_length', 'cds_length', 'pep_length', 'pep_start_on_cdna', 'pep_end_on_cdna', 'number_of_Us', 'nU', 'U_pos_on_pep', 'U_pos_on_cdna']) + '\n')
        for line in inFile:
            line = line.strip()
            fields = line.split('\t')
            cdna = fields[5]
            cds = fields[6]
            pep = fields[7]
            cdna_length = len(cdna)
            cds_length = len(cds)
            pep_length = len(pep)
            pep_start_on_cdna = -1
            pep_end_on_cdna = -1
            number_of_Us = -1
            number_U = -1
            U_pos_on_pep = -1
            U_pos_on_cdna = -1
            if cdna.find(cds) != -1 and cds_length == pep_length * 3 + 3:
                pep_start_on_cdna = cdna.find(cds) + 1
                pep_end_on_cdna = pep_start_on_cdna + cds_length - 1 - 3
                number_of_Us = pep.count('U')
                if number_of_Us:
                    n = 0
                    for i,c in enumerate(pep):
                        if c == 'U':
                            n += 1
                            number_U = n
                            U_pos_on_pep = i + 1
                            U_pos_on_cdna = pep_start_on_cdna + i * 3
                            ouFile.write(line + '\t' + '\t'.join([str(x) for x in [cdna_length, cds_length, pep_length, pep_start_on_cdna, pep_end_on_cdna, number_of_Us, number_U, U_pos_on_pep, U_pos_on_cdna]]) + '\n')
                else:
                    ouFile.write(line + '\t' + '\t'.join([str(x) for x in [cdna_length, cds_length, pep_length, pep_start_on_cdna, pep_end_on_cdna, number_of_Us, number_U, U_pos_on_pep, U_pos_on_cdna]]) + '\n')
            else:
                ouFile.write(line + '\t' + '\t'.join([str(x) for x in [cdna_length, cds_length, pep_length, pep_start_on_cdna, pep_end_on_cdna, number_of_Us, number_U, U_pos_on_pep, U_pos_on_cdna]]) + '\n')
    
        inFile.close()
        ouFile.close()

    def get_feature_seq(self, in_file='Ensembl_Protein2Transcript.txt.gz', flank=50, pad=1000):
        inFile = gzip.open(in_file.replace('.txt.gz', '_Upos.txt.gz'), 'rt')
        ouFile = gzip.open(in_file.replace('.txt.gz', '_Upos_FeatureSeq.txt.gz'), 'wt')
        head = inFile.readline().strip()
        ouFile.write(head + '\tclass\tfeatureSeq\n')
        for line in inFile:
            line = line.strip()
            fields = line.split('\t')
            cdna = fields[5].upper()
            cdna_pad = 'N' * pad + cdna + 'N' * pad
            U_pos_on_cdna = int(fields[17])
            stop_pos_on_cdna = int(fields[13]) + 1
            if int(fields[12]) != -1:
                if int(fields[14]) == 0:
                    cls = 'negative'
                    pos = stop_pos_on_cdna + pad
                else:
                    cls = 'positive'
                    pos = U_pos_on_cdna + pad
                seq = cdna_pad[pos - flank - 1 : pos + flank - 1]
                ouFile.write(line + '\t' + cls + '\t' + seq + '\n')
    
        inFile.close()
        ouFile.close()

    def down_sampling_negtive_samples(self, in_file='Ensembl_Protein2Transcript.txt.gz', n_fold=100):
        D = {}
        inFile = gzip.open(in_file.replace('.txt.gz', '_Upos_FeatureSeq.txt.gz'), 'rt')
        head = inFile.readline()
        for line in inFile:
            line = line.strip()
            fields = line.split('\t')
            sp = fields[0]
            cls = fields[-2]
            if cls == 'positive':
                D.setdefault(sp, [])
        inFile.close()

        L = []
        S = set()
        inFile = gzip.open(in_file.replace('.txt.gz', '_Upos_FeatureSeq.txt.gz'), 'rt')
        ouFile = gzip.open(in_file.replace('.txt.gz', f'_Upos_FeatureSeq_DownSampling{n_fold}.txt.gz'), 'wt')
        head = inFile.readline()
        ouFile.write(head)
        for line in inFile:
            line = line.strip()
            fields = line.split('\t')
            sp = fields[0]
            cls = fields[-2]
            seq = fields[-1]
            if cls == 'positive':
                L.append(fields)
            else:
                if sp in D:
                    D[sp].append(fields)
        inFile.close()

        for item in L:
            seq = item[-1]
            if seq not in S:
                S.add(seq)
                ouFile.write('\t'.join(item) + '\n')

                sp = item[0]
                if sp in D:
                    L = D[sp]
                    ## sort by distance to end_cdna and start_cdna within the same species
                    L = sorted(L, key = lambda x: np.linalg.norm(np.array([int(x[9]) - int(x[13]), int(x[13])]) - np.array([int(item[9]) - int(item[17]), int(item[17])])))
                    for item in L[0:n_fold]:
                        ouFile.write('\t'.join(item) + '\n')
        ouFile.close()

if __name__ == '__main__':
    dp = DataProcessor()
    #dp.pep2cds2cdna()
    #dp.get_pos_of_U()
    #dp.get_feature_seq()
    dp.down_sampling_negtive_samples()
