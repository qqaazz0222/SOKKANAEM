"""Build the local MDPI review PDF with isolated, pinned typesetting tools."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record


def main():
    sub = Path('paper/submission')
    gs = Path('work_dirs/paper_tools/ghostscript/bin/gs')
    tectonic = Path('work_dirs/paper_tools/tectonic-0.15.0/tectonic')
    (sub/'converted').mkdir(exist_ok=True)
    for logo in ('logo-mdpi', 'logo-updates'):
        subprocess.run([str(gs), '-dSAFER', '-dBATCH', '-dNOPAUSE', '-dEPSCrop',
            '-sDEVICE=pdfwrite', '-sOutputFile='+str(sub/'converted'/f'{logo}.pdf'),
            str(sub/'Definitions'/f'{logo}.eps')], check=True)
    subprocess.run([str(tectonic), '--untrusted', '--keep-logs', str(sub/'manuscript.tex')], check=True)
    log = (sub/'manuscript.log').read_text()
    errors = [s for s in log.splitlines() if s.startswith('! ') or any(x in s for x in
        ('Overfull', 'undefined references', 'undefined citations', 'Missing character:'))]
    if errors:
        raise ValueError('PDF requires inspection: '+repr(errors))
    result = dict(pdf=file_record(sub/'manuscript.pdf'), log=file_record(sub/'manuscript.log'),
        tools=[file_record(gs), file_record(tectonic)],
        source=[file_record(p) for p in sorted(sub.glob('*.tex'))],
        converted_logos=[file_record(p) for p in sorted((sub/'converted').glob('*.pdf'))],
        critical_layout_or_reference_errors=errors, author_review_required=True,
        warning='Noncritical package/underfull-box warnings may remain; inspect the PDF visually.')
    Path('work_dirs/paper_closeout/pdf_build.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Built', sub/'manuscript.pdf')


if __name__ == '__main__':
    main()
