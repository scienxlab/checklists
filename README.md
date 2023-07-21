# checklists

A collection of checklists and decision trees for scientists and engineers.

The checklists are maintained as [Markdown](https://www.markdownguide.org/) files for convenience, but can be rendered in other formats using conversion tools like [pandoc](https://pandoc.org/). For example, to render the `machine-learning-projects` checklist as a PDF, you can do this:

```shell
pandoc -t html --css style.css -V margin-top=7 -V margin-right=10 -V margin-bottom=7 -V margin-left=10 machine-learning-projects.md -o out.pdf
```

You can install `pandoc` on Ubuntu with `sudo apt install pandoc`; builds exist for most distros. To use the command above, you may also need to install the `wkhtmltopdf` tool as well.

In the near future, we will provide PDFs via links from this repository.


## Contents

- **[machine-learning-projects](./machine-learning-projects.md)** &mdash; Annotated checklist for people running or reviewing machine learning projects.


## Coming soon

- Reproducible scientific project ~~checklist~~wishlist
- Machine learning project planning checklist
- Scientific presentation checklist
- Computing course checklist
- Hackathon planning checklist
- Which programming language should I learn first?
- So you want to build a Python app
