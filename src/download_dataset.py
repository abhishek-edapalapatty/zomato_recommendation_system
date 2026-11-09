"""Download the source dataset used by the referenced Kaggle notebook."""
import shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data'/'zomato.csv'
SLUG='himanshupoddar/zomato-bangalore-restaurants'

def main():
    if shutil.which('kaggle') is None:
        raise SystemExit('Install Kaggle CLI: python -m pip install kaggle. Then configure your Kaggle API token and rerun this script.')
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run(['kaggle','datasets','download','-d',SLUG,'-p',folder,'--force'],check=True)
        archives=list(Path(folder).glob('*.zip'))
        if not archives: raise SystemExit('Kaggle download completed but no ZIP archive was found.')
        with zipfile.ZipFile(archives[0]) as archive:
            candidates=[n for n in archive.namelist() if Path(n).name.lower()=='zomato.csv']
            if not candidates: raise SystemExit('The archive did not contain zomato.csv. Inspect the Kaggle dataset files and update this loader.')
            DEST.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(candidates[0]) as src, DEST.open('wb') as dst: shutil.copyfileobj(src,dst)
    print(f'Dataset saved to {DEST}')
if __name__=='__main__': main()
