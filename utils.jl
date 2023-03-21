using Franklin: LxCom, stent

"""
    formatdate(date::Date, format=globvar(:date_format))

Format the date object as a html <time> tag according to the given format.
"""
function formatdate(date::Date, format=globvar(:date_format))
    """<time datetime="$date">$(Dates.format(date, format))</time>"""
end

"""
    getvar(page, var; default=nothing)

Get `var` from `page`, using the simplier `Franklin.locvar` if possible.
"""
function getvar(page, var; default=nothing)
    (page == locvar(:fd_rpath)) ?
        locvar(var; default) : pagevar(page, var; default)
end

"""
    robust_title(page)

Get the title field for `page`, defaulting to the path if not defined.
"""
robust_title(page) = getvar(page, :title; default="/$page/")

"""
    robust_date(page, format=globvar(:date_format))

Get the date field for `page`, defaulting to date of creation if not defined.
"""
function robust_date(page, format=globvar(:date_format))
    date = getvar(page, :date;
                  default=Date(Dates.unix2datetime(stat(page * ".md").ctime))
                 )
    formatdate(date, format)
end

"""
    modification_date()

Get the time of last modification for the current page.
"""
function modification_date()
    """<time datetime="$(locvar(:fd_mtime_raw))">$(locvar(:fd_mtime))</time>"""
end

"""
    write_header!(io, page; rss=true)

Render the metadata about the blog post `page` to `io`.
"""
function write_header!(io, page; rss=true)
    # short description
    description = pagevar(page, :rss_description)
    if !isnothing(description) && rss
        write(io, "<p>$description</p>")
    end
    # date
    date = robust_date(page)
    write(io, """<span class="post-meta">$date</span>""")
    # tags
    tags = pagevar(page, :tags; default=String[])
    if length(tags) > 0
        tag_path = globvar(:tag_page_path)
        write(io, """<span class="post-tags">&nbsp; &middot;""")
        for tag in tags
            tag_url = "/$tag_path/$tag/"
            write(io, """&nbsp; <a href="$tag_url"><b>#</b> $tag</a>""")
        end
        write(io, "</span>")
    end
end

"""
    hfun_assets()

Get a path to the assets directory for the current page.
"""
hfun_assets() = "/assets$(get_url(locvar(:fd_rpath)))"[begin:end - 1]

"""
    hfun_makeheader()

Make the header list for the website.
"""
function hfun_makeheader()
    # if tag page, default to highlighting blog
    path = isempty(locvar(:fd_tag)) ? locvar(:fd_rpath) : "blog/index.md"
    current_url = get_url(path)
    io = IOBuffer()
    write(io, "<ul>")
    for (url, name) in globvar(:headers)
        is_active = (url == current_url) ? "active" : ""
        write(io, """<li><a href="$url" class="$is_active">$name</a></li>\n""")
    end
    write(io, "</ul>")
    return String(take!(io))
end

"""
    hfun_pagesource()

Return the page source, ignoring automatically generated pages.
"""
function hfun_pagesource()
    # early exit for tag pages
    !isempty(locvar(:fd_tag)) && return ""
    repo = "$(globvar(:git_repo))/$(locvar(:fd_rpath))"
    return (
        "<a href=\"$(repo)\">Page source</a>." *
        (isempty(hfun_lastupdated()) ? "" : " ")
    )
end

"""
    hfun_lastupdated()

Return the modification time, ignoring automatically generated pages.
"""
function hfun_lastupdated()
    date = locvar(:fd_mtime_raw)
    # early exit for tag pages
    !isempty(locvar(:fd_tag)) && return ""
    url = get_url(locvar(:fd_rpath))
    exclude = globvar(:footer_exclude)
    (in(url, exclude)) ?  "" : "Last updated: $(formatdate(date))."
end

"""
    hfun_custom_taglist()

Generate a custom tag list.

See: https://tlienart.github.io/FranklinTemplates.jl/templates/basic/menu3/
"""
function hfun_custom_taglist()
    # --------------------------------------------------------------
    # Part 1: Retrieve all pages associated with the tag & sort them
    # --------------------------------------------------------------
    # retrieve the tag string
    tag = locvar(:fd_tag)
    # recover the relative paths to all pages that
    # have that tag, these are paths like /blog/page1
    rpaths = globvar(:fd_tag_pages)[tag]
    # you might want to sort these pages by chronological order
    # you could also only show the most recent 5 etc...
    sort!(rpaths, by=robust_date, rev=true)

    # ---------------------------------
    # Part 2: Write the HTML to plug in
    # ---------------------------------
    # instantiate a buffer in which we will
    # write the HTML to plug in the tag page
    io = IOBuffer()
    write(io, """<div class="tagged-posts clean-table"><table><tbody>\n""")
    # go over all paths
    for rpath in rpaths
        write(io, "<tr>")
        # recover the url corresponding to the rpath
        url = get_url(rpath)
        title, date = robust_title(rpath), robust_date(rpath)
        # write some appropriate HTML
        write(io, """<th scope="row">$date</th>""")
        write(io, """<td><a href="/$rpath/">$title</a></td>""")
        write(io, "</tr>\n")
    end
    # finish the HTML
    write(io, "</tbody></table></div>")
    # return the HTML string
    return String(take!(io))
end

"""
    hfun_gettags()

Render the list of tags for blog posts.
"""
function hfun_gettags()
    path = globvar(:tag_page_path)
    tags = locvar(:displaytags)
    io = IOBuffer()
    write(io, "<ul>\n")
    for (i, tag) in enumerate(tags)
        url = "/$path/$tag/"
        write(io, """<li><a href="$url"><b>#</b> $tag</a>""")
        i != length(tags) && write(io, "<p>&nbsp; &middot; &nbsp;</p>")
        write(io, "</li>\n")
    end
    write(io, "</ul>")
    return String(take!(io))
end

"""
    hfun_getposts()

Render the list of blog posts in reverse chronological order.

See: https://franklinjl.org/demos/#007_delayed_hfun
"""
@delay function hfun_getposts()
    io = IOBuffer()
    write(io, "<ul>\n")
    for post in sort(readdir("blog"; join=true), by=robust_date, rev=true)
        post == "blog/index.md" && continue
        write(io, "<li>")
        url, title = get_url(post), robust_title(post)
        write(io, """<h3><a href="$url">$title</a></h3>""")
        write_header!(io, post)
        write(io, "</li>\n")
    end
    write(io, "</ul>")
    return String(take!(io))
end

"""
    hfun_maketitle()

Make the title for blog posts.
"""
function hfun_maketitle()
    io = IOBuffer()
    post = locvar(:fd_rpath)
    title = robust_title(post)
    write(io, "<h1>$title</h1>\n")
    write_header!(io, post; rss=false)
    return String(take!(io))
end

"""
    lx_news(com, _)

Get the `n` most recent news entries.
"""
function lx_news(com, _)
    n = parse(Int64, stent(com.braces[1]))
    io = IOBuffer()
    i = -1
    open("news.md") do news
        for line in eachline(news)
            if line == "@@news,clean-table,no-header"
                i = 0
            end
            i >= 0 && (i += 1)
            1 <= i <= n + 3 && write(io, line, "\n")
        end
    end
    write(io, "@@", "\n")
    return String(take!(io))
end

"""
    lx_makecard(com, _)

Make a card for the given project page.
"""
function lx_makecard(com, _)
    page = stent(com.braces[1])
    path = "projects/$page"
    image = pagevar(path, :preview_image)
    title = pagevar(path, :title)
    description = pagevar(path, :rss_description)
    return "\\card{$page}{$image}{$title}{$description}"
end

"""
    lx_cite(lxc::LxCom, _)

Wrap citation with brackets, e.g. [1]. Use \\citet for the original behavior.

See also: [\\citet](@ref), [\\citep](@ref) as defined in
https://github.com/tlienart/Franklin.jl/blob/master\
/src/converter/latex/hyperrefs.jl.
"""
function Franklin.lx_cite(lxc::LxCom, _)
    Franklin.form_href(lxc, "BIBR"; parens="["=>"]", class="bibref")
end

"""
    lx_cref(lxc::LxCom, _)

Cross-reference like the cleveref package. Currently only supports equations.

See also: [\\eqref](@ref).
"""
function lx_cref(lxc::LxCom, _)
    io = IOBuffer()
    write(io, "~~~")
    write(io, Franklin.form_href(lxc, "EQR"; class="eqref cref"))
    write(io, "~~~")
    return String(take!(io))
end

"""
    lx_bibliography(com, _)

Render the BibTeX bibliography with `pandoc`.

See: https://ctroupin.github.io/posts/2019-12-19-bibtex-markdown/
"""
function lx_bibliography(com, _)
    bib = stent(com.braces[1])
    path = "/assets$(get_url(locvar(:fd_rpath)))"[begin:end - 1]
    """
    @@references
    ## References
    [BibTeX]($path/$bib.bib)

    \\textinput{$path/$bib.md}
    @@
    """
end

"""
    env_wrap(com, _)

Wrap the contents in a html tag without spurious <p> and </p>'s.

See: [convert_md](@ref), [reprocess](@ref),
https://github.com/tlienart/Franklin.jl/issues/677
"""
function env_wrap(com, _)
    tag_data = stent(com.braces[1])
    tag, data... = split(tag_data, " ")
    data = (length(data) > 0) ? " $(join(data, ' '))" : ""
    content = stent(com)
    lxdefs = collect(values(Franklin.GLOBAL_LXDEFS))
    # https://github.com/tlienart/Franklin.jl/blob/4ba6d9020367468bfb77b5bde9eabb2648ab8a21/src/converter/markdown/blocks.jl#L35-L37
    parsed = Franklin.reprocess(content, lxdefs;
                                nostripp=true) |> Franklin.simplify_ps
    return "~~~<$tag$data>$parsed</$tag>~~~"
end

