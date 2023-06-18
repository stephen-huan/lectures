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
    "utils/",
    "misc/diffy-geosystems/images/README.md",
]

# RSS (the website_{title, descr, url} must be defined to get RSS)
generate_rss = false
website_title = "Lectures"
website_descr = "Lecture notes on computer science and mathematics."
website_url   = "https://lectures.cgdct.moe/"
# prepath = "lectures"

# git repo for page source
git_repo = "https://github.com/stephen-huan/lectures/blob/master"

footer_text = "単純明快。"

# footer exclude
footer_exclude = Set(
    ["/404/"]
)
+++
