import os
import gzip
import glob
import subprocess

class DataDownloader:
    def __init__(self):
        self.download_cmd = '''
SERVER="{}"
REMOTE_DIR="{}"
LOCAL_DIR="{}"
mkdir -p "$LOCAL_DIR"
lftp $SERVER << EOF
cd $REMOTE_DIR
mirror --include-glob=*pep*.gz --parallel=2 "$REMOTE_DIR" "$LOCAL_DIR"
mirror --include-glob=*cds*.gz --parallel=2 "$REMOTE_DIR" "$LOCAL_DIR"
mirror --include-glob=*cdna*.gz --parallel=2 "$REMOTE_DIR" "$LOCAL_DIR"
quit
EOF
'''
    def download_data(self, from_ensembl=True, ensemble_version=111, from_ensemblgenomes=True, ensemblgenomes_version=62, out_dir='data'):
        if os.path.exists(out_dir) is False:
            os.makedirs(out_dir)

        if from_ensembl:
            server = "ftp.ensembl.org"
            remote_dir = f"/pub/release-{ensemble_version}/fasta/"
            cmd = self.download_cmd.format(server, remote_dir, f'{out_dir}/ensembl')
            subprocess.run(cmd, shell=True, check=True)

        if from_ensemblgenomes:
            for division in ['bacteria', 'fungi', 'metazoa', 'plants', 'protists']:
                server = "ftp.ensemblgenomes.org"
                remote_dir = f"/pub/release-{ensemblgenomes_version}/{division}/fasta/"
                cmd = self.download_cmd.format(server, remote_dir, f'{out_dir}/{division}')
                subprocess.run(cmd, shell=True, check=True)

        self.fasta_concat()

    def fasta_concat(self, in_dir='data'):
        fasta_files = glob.glob(f'{in_dir}/**/*.fa.gz', recursive=True)
        for in_file in fasta_files:
            if not in_file.endswith('.OneLine.fa.gz'):
                try:
                    print(f'Processing {in_file} ...')
                    self._OneLine(in_file)
                except Exception as e:
                    print(f'Error processing {in_file}: {e}')

    def _OneLine(self, in_file):
        out_file = in_file.split('.fa.gz')[0] + '.OneLine.fa.gz'
        ouFile = gzip.open(out_file, 'wt')
        D = {}
        L = []

        inFile = gzip.open(in_file, 'rt')
        for line in inFile:
            line = line.strip()
            if line.find('>') == 0:
                k = line
                D.setdefault(k, [])
                L.append(k)
            else:
                D[k].append(line)
        inFile.close()

        for k in L:
            seq = ''.join(D[k])
            ouFile.write(k + '\n')
            ouFile.write(seq + '\n')
        ouFile.close()

if __name__ == '__main__':
    dd = DataDownloader()
    dd.download_data()
