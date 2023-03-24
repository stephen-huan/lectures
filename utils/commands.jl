using Franklin: GLOBAL_LXDEFS, lxd, lxe, subs

# https://github.com/tlienart/Franklin.jl/blob/master/\
# src/converter/latex/objects.jl
const commands = [
    # text formatting
    lxd("bold", 1, raw"""
        \begin{wrap}{span class="font-weight-bold"}#1\end{wrap}"""
    ),
    lxd("emph", 1, raw"""_!#1_"""),
    lxd("href", 2, raw"""\begin{wrap}{a href="!#1"}#2\end{wrap}"""),
    lxd("url",  1, raw"""\href{!#1}{~~~#1~~~}"""),
    lxd("chapter",       1, raw"""\begin{wrap}{h1}#1\end{wrap}"""),
    lxd("section",       1, raw"""\begin{wrap}{h2}#1\end{wrap}"""),
    lxd("subsection",    1, raw"""\begin{wrap}{h3}#1\end{wrap}"""),
    lxd("subsubsection", 1, raw"""\begin{wrap}{h4}#1\end{wrap}"""),
    lxd("paragraph",     1, raw"""\begin{wrap}{h5}#1\end{wrap}"""),
    lxd("subparagraph",  1, raw"""\begin{wrap}{h6}#1\end{wrap}"""),
    # images
    lxd("includegraphics", 4, raw"""
        ~~~<img alt="!#1" src="!#2" width="!#3" height="!#4">~~~"""
    ),
    lxd("caption", 1, raw"""\begin{wrap}{figcaption}#1\end{wrap}"""),
    lxd("figpreview", 5, raw"""
        \begin{wrap}{a href="!#5"}
            \includegraphics{!#1}{{{assets}}/!#2}{!#3}{!#4}
        \end{wrap}"""
    ),
]

const environments = [
    # images
    lxe("figure", 0, raw"""\begin{wrap}{figure}""" => raw"""\end{wrap}"""),
    # columns
    lxe("columns", 0, raw"""
        \begin{wrap}{div class="row"}""" => raw"""
        ~~~<div class="column-end"></div>~~~\end{wrap}"""
    ),
    lxe("column", 2,
        raw"""\begin{wrap}{div class="#1 #2"}""" => raw"""\end{wrap}"""
    ),
]

for store in (commands, environments), (name, def) in store
    GLOBAL_LXDEFS[name] = def
end

