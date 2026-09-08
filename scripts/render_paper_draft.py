"""Synchronize the full Markdown paper from the integrated LaTeX source, with PNG figures."""
import json
import html
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paper_closeout_study import verify_files
from sokkanaem.protocol import file_record

SUB = Path('paper/submission')
PANDOC = Path('work_dirs/paper_tools/document_python/pypandoc/files/pandoc')


def argument(source, start):
    """Read a balanced TeX argument, preserving escaped braces."""
    if source[start] != '{':
        raise ValueError('expected opening brace')
    depth, i = 1, start+1
    while i < len(source):
        if source[i] == '\\':
            i += 2
            continue
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                return source[start+1:i], i+1
        i += 1
    raise ValueError('unclosed TeX argument')


def command_value(source, command):
    match = re.search(r'\\'+command+r'\s*\{', source)
    if not match:
        raise ValueError('missing command '+command)
    return argument(source, match.end()-1)[0]


def expand_inputs(source, inputs):
    def expand(match):
        path = SUB/match[1]
        if path.suffix != '.tex':
            path = path.with_suffix('.tex')
        inputs.add(path)
        return expand_inputs(path.read_text(), inputs)
    return re.sub(r'\\input\{([^}]+)\}', expand, source)


def prepare(source):
    inputs = {SUB/'manuscript.tex', SUB/'generated_macros.tex', SUB/'author_details.tex'}
    title, abstract = command_value(source, 'Title'), command_value(source, 'abstract')
    body = source.split('\\begin{document}', 1)[1].split('\\end{document}', 1)[0]
    body = expand_inputs(body, inputs)
    body = body.replace('\\begin{adjustwidth}{-\\extralength}{0cm}', '').replace('\\end{adjustwidth}', '')
    # Turn MDPI-only declarations into ordinary sections without dropping their content.
    declaration_names = dict(supplementary='Supplementary materials', authorcontributions='Author contributions',
        funding='Funding', institutionalreview='Institutional review', informedconsent='Informed consent',
        dataavailability='Data availability', acknowledgments='Acknowledgments and AI assistance',
        conflictsofinterest='Conflicts of interest')
    first = True
    for name, heading in declaration_names.items():
        match = re.search(r'\\'+name+r'\s*\{', body)
        if match:
            value, end = argument(body, match.end()-1)
            replacement = ('\\section{Declarations}\n' if first else '')+'\\subsection{'+heading+'}\n'+value+'\n'
            body = body[:match.start()] + replacement + body[end:]
            first = False
    body = re.sub(r'\\appendixtitles\{[^}]*\}|\\appendixstart|\\appendix\b', '', body)
    body = body.replace('\\section{Analytic Proofs}', '\\section{Appendix A. Analytic Proofs}')
    body = body.replace('\\section{Reproduction Details and Evidence Boundaries}',
                        '\\section{Appendix B. Reproduction Details and Evidence Boundaries}')
    # Bibliography is explicit, so no implicit citeproc database or unresolved cite keys remain.
    bib = body.split('\\begin{thebibliography}{99}', 1)[1].split('\\end{thebibliography}', 1)[0]
    entries = re.findall(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\Z)', bib, flags=re.S)
    refs = {key: i+1 for i, (key, _) in enumerate(entries)}
    bibliography = '\\section{References}\n'+''.join(
        '\\hypertarget{ref-'+key+'}{}\\paragraph{['+str(refs[key])+']} '+value+'\n' for key, value in entries)
    body = re.sub(r'\\reftitle\{References\}\s*\\begin\{thebibliography\}\{99\}.*?\\end\{thebibliography\}',
                  lambda _: bibliography, body, flags=re.S)
    body = re.sub(r'\\cite\{([^}]+)\}', lambda m: ', '.join(
        '\\hyperlink{ref-'+key+'}{['+str(refs[key])+']}' for key in m[1].split(',')), body)
    # Preserve theorem/proof prose that a generic LaTeX reader would otherwise omit.
    counts = {'Theorem': 0, 'Proposition': 0}
    labels = {'app:proofs': 'A', 'app:repro': 'B'}
    for kind in ('table', 'figure'):
        for i, block in enumerate(re.findall(r'\\begin\{'+kind+r'\}.*?\\end\{'+kind+r'\}', body, re.S), 1):
            for key in re.findall(r'\\label\{([^}]+)\}', block):
                labels[key] = str(i)
    def theorem(match):
        kind, caption, label = match[1], match[2], match[3]
        counts[kind] += 1
        number = str(counts[kind])
        if label:
            labels[label] = number
        return '\\paragraph{'+kind+' '+number+' ('+caption+').}\n'
    body = re.sub(r'\\begin\{(Theorem|Proposition)\}\[([^]]+)\](?:\\label\{([^}]+)\})?', theorem, body)
    body = re.sub(r'\\end\{(?:Theorem|Proposition)\}', '', body)
    body = body.replace('\\begin{proof}', '\\paragraph{Proof.}').replace('\\end{proof}', '\\hfill $\\square$\n')
    eq_names = {'eq:gate': 'the binary-gate equation', 'eq:bound': 'the defect-propagation bound',
                'eq:cache': 'the first-block cache bound'}
    body = re.sub(r'\\eqref\{([^}]+)\}', lambda m: '('+eq_names[m[1]]+')', body)
    body = re.sub(r'\\ref\{([^}]+)\}', lambda m: labels[m[1]], body)
    body = re.sub(r'\\label\{[^}]+\}', '', body)
    # Figure / table captions retain an explicit number in a standalone Markdown rendering.
    for kind in ('figure', 'table'):
        counter = [0]
        def caption_number(match):
            counter[0] += 1
            return match[0].replace('\\caption{', '\\caption{'+kind.title()+' '+str(counter[0])+'. ', 1)
        body = re.sub(r'\\begin\{'+kind+r'\}.*?\\end\{'+kind+r'\}', caption_number, body, flags=re.S)
    prefix = (SUB/'generated_macros.tex').read_text()+'\n\\newcommand{\\norm}[1]{\\lVert#1\\rVert}\n'
    authors = (SUB/'author_details.tex').read_text()
    author_text = '\\section*{Authors}\n'+command_value(authors, 'Author')+'\\par\n'
    author_text += command_value(authors, 'address')+'\\par\n'+command_value(authors, 'corres')+'\n'
    return title, prefix+author_text+'\\section*{Abstract}\n'+abstract+'\n'+body, inputs, refs


def main():
    for name in ('results', 'figures'):
        audit = json.loads(Path(f'work_dirs/paper_revision_r3/{name}.json').read_text())
        verify_files([audit['generator'], *audit['inputs'], *audit['artifacts']])
    title, tex, inputs, refs = prepare((SUB/'manuscript.tex').read_text())
    result = subprocess.run([str(PANDOC), '-f', 'latex', '-t', 'gfm+tex_math_dollars', '--wrap=none'],
                            input=tex, text=True, capture_output=True, check=True)
    md = result.stdout
    # Pandoc may emit an HTML image rather than a Markdown image; both share the path spelling.
    md = re.sub(r'figures/([A-Za-z0-9_]+)\.pdf', r'submission/figures/\1.png', md)
    def figure_markdown(match):
        path, caption = match[1], html.unescape(match[2])
        caption = re.sub(r'<[^>]+>', '', caption).strip()
        short = caption.split('. ', 2)[:2]
        return '!['+'. '.join(short).replace('[', '').replace(']', '')+']('+path+')\n\n'+caption+'\n'
    md = re.sub(r'<figure>\s*<embed src="([^"]+)"\s*/>\s*<figcaption>(.*?)</figcaption>\s*</figure>',
                figure_markdown, md, flags=re.S)
    md = re.sub(r'\$`(.*?)`\$', lambda m: '$'+m[1]+'$', md, flags=re.S)
    md = re.sub(r'``` math\n(.*?)\n```', lambda m: '\n$$\n'+m[1]+'\n$$\n', md, flags=re.S)
    md = re.sub(r'^(#{1,5}) ', r'\1# ', md, flags=re.M)
    md = re.sub(r'\{#([^}]+)\}', '', md)
    header = ('# '+title+'\n\n'
        '> 모델 중심 개정 원고 · 2026-09-08. `paper/submission/manuscript.tex`에서 동기화한 전체 Markdown판입니다.\n'
        '> [PDF](submission/manuscript.pdf) · [40항목 self-revision 대응표](self-revision/r3/revision-status.md) · '
        '[정정 기록](submission/REVISION_NOTES.md) · [과거 초안](draft_legacy_20260907.md).\n'
        '> 저자 정보·독립 검토·권리 확인은 대기 중이며 투고 완료본이 아닙니다. 그림은 저장된 실측/평가 결과와 구현에서 생성했습니다.\n\n'
        '')
    md = header+md
    if any(token in md for token in ('\\input{', '\\FinalNative', '\\begin{Theorem}', '\\cite{', '<embed')):
        raise ValueError('unconverted source command in Markdown')
    # Fail closed if an input image or a major manuscript section disappeared in conversion.
    images = sorted(set(re.findall(r'submission/figures/[\w]+\.png', md)))
    if len(images) != 7 or not all((Path('paper')/p).is_file() for p in images):
        raise ValueError('expected seven existing manuscript images')
    for required in ('Abstract', 'Conclusions', 'Declarations', 'Appendix A.', 'Appendix B.',
                     'References', 'Shared-input defect propagation', 'Conditional pipeline speedup'):
        if required not in md:
            raise ValueError('lost section: '+required)
    dest = Path('paper/draft.md')
    dest.write_text(md)
    artifact = dict(generator=file_record(__file__), pandoc=file_record(PANDOC),
        pandoc_version=subprocess.check_output([str(PANDOC), '--version'], text=True).splitlines()[0],
        inputs=[file_record(p) for p in sorted(inputs)]+[file_record('work_dirs/paper_revision_r3/figures.json'),
            file_record('work_dirs/paper_qualitative_20260908/render.json')],
        artifacts=[file_record(dest)], images=images, reference_count=len(refs),
        full_manuscript=True, author_review_required=True, conversion_warnings=result.stderr)
    Path('work_dirs/paper_revision_r3/draft.json').write_text(json.dumps(artifact, indent=2)+'\n')
    print('Updated paper/draft.md:', len(md.split()), 'words;', len(images), 'images;', len(refs), 'references')
    if result.stderr:
        print(result.stderr)


if __name__ == '__main__':
    main()
