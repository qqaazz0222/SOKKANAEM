from pathlib import Path
import re
import pytest
from scripts.render_paper_draft import argument, prepare


def test_balanced_tex_arguments():
    value, end = argument(r'{outer {inner} \{literal\}}suffix', 0)
    assert value == r'outer {inner} \{literal\}'
    assert r'{outer {inner} \{literal\}}suffix'[end:] == 'suffix'
    with pytest.raises(ValueError):
        argument('{unclosed', 0)


def test_conversion_keeps_proofs_and_declarations():
    title, tex, inputs, refs = prepare(Path('paper/submission/manuscript.tex').read_text())
    assert 'State Preservation' in title
    assert r'\paragraph{Theorem 1 (Shared-input defect propagation).}' in tex
    assert r'\subsection{Funding}' in tex
    assert r'\section{References}' in tex
    assert len(refs) == 14
    assert Path('paper/submission/author_details.tex') in inputs


def test_current_markdown_images_and_full_sections():
    md = Path('paper/draft.md').read_text()
    images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', md)
    assert len(images) == 5
    assert all((Path('paper')/p).is_file() for p in images)
    assert '<embed' not in md and '\\FinalNative' not in md
    assert 'Appendix A.' in md and 'Appendix B.' in md and '## References' in md
