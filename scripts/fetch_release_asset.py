"""Download and verify a logical release asset, joining small transport parts."""
import argparse,json
from pathlib import Path
from reproduce_reported_results import download
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();info=json.loads(a.manifest.read_text(encoding='utf-8'))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    download(info['base_url']+info['filename'],a.output,info)
    print('Verified:',a.output)
if __name__=='__main__':main()
