<!--
Add here global page variables to use throughout your website.
-->
+++
author = "Stephen Huan"
date_format= "yyyy-mm-dd"
mintoclevel = 2

# Add here files or directories that should be ignored by Franklin, otherwise
# these files might be copied and, if markdown, processed by Franklin which
# you might not want. Indicate directories by ending the name with a `/`.
# Base files such as LICENSE.md and README.md are ignored by default.
ignore = [
    "Project.toml",
    "Manifest.toml",
    "node_modules/",
    "package-lock.json",
    "package.json",
    ".prettierrc.json",
    ".prettierignore",
    "bin/",
    "misc/diffy-geosystems/images/README.md",
]

# RSS (the website_{title, descr, url} must be defined to get RSS)
generate_rss = true
website_title = "Lectures"
website_descr = "Lectures on computer science and mathematics"
website_url   = "https://lectures.cgdct.moe/"
# prepath = "lectures"

# character for the icon
icon = "浣"

# git repo for page source
git_repo = "https://github.com/stephen-huan/lectures/blob/master"

# footer exclude
footer_exclude = Set(
    ["/404/"]
)
+++

<!--
Add here global LaTeX commands to use throughout your pages.
-->
\newcommand{\chapter      }[1]{\begin{wrap}{h1}#1\end{wrap}}

