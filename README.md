# PU-Boost arXiv / conference-style LaTeX project

Final layout revision includes:
- title text block and the two title rules at the same width;
- two-column conference-style body;
- Figures 3, 4, and 5 as one-column figures with one-column captions;
- Figure 8 arranged as an equal-height side-by-side SHAP summary + dependence-panel group;
- consistent conference-style type hierarchy and compact display/float spacing;
- complete formulas retained;
- all author emails shown; only Huanqi Wu carries the corresponding-author asterisk;
- three-line tables and coordinated figure/table numbering.

Compile with pdfLaTeX twice:

    pdflatex main.tex
    pdflatex main.tex

Revision final_v2: institution names removed directly under author names (email only); abstract and keywords use the same base font size as the body text.
