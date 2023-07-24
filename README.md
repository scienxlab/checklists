# ☑️ checklists

A collection of checklists and decision trees for scientists and engineers.

The checklists are maintained as [Markdown](https://www.markdownguide.org/) files for convenience, but can be rendered in other formats using conversion tools like [pandoc](https://pandoc.org/). For example, to render the `machine-learning-review` checklist as a PDF, you can do this:

```shell
pandoc -t html --css style.css \
       -V margin-top=7 -V margin-right=10 \
       -V margin-bottom=7 -V margin-left=10 \
       machine-learning-review.md -o out.pdf
```

You can install `pandoc` on Ubuntu with `sudo apt install pandoc`; builds exist for most distros. To use the command above, you may also need to install the `wkhtmltopdf` tool as well.

In the near future, we will provide PDFs via links from this repository.


## Contents

- [`machine-learning-review`](./machine-learning-review.md) &mdash; Annotated checklist for people running or reviewing machine learning projects.
- [`machine-learning-planning`](./machine-learning-planning.md) &mdash; Annotated checklist specifically for earth scientists planning machine learning projects.
- [`reproducible-science-projects`](./reproducible-science-projects.md) &mdash; An aspirational wishlist for open source and reproducible scientific repos.


## Coming soon

- Open data checklist
- Benchmark dataset checklist
- Scientific presentation checklist
- Computing course checklist
- Hackathon planning checklist
- Which programming language should I learn first?
- So you want to build a Python app
