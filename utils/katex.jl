using Franklin: GLOBAL_LXDEFS, lxd, lxe, subs

"""
    make_def(name)

Make a definition-like environment with the given name.
"""
function make_def(name)
    return """
        \\begin{wrap}{div class="$(lowercase(name))"}
        **$(uppercasefirst(name))**.""" => raw"""
        \end{wrap}"""
end

"""
    starmacro()

Make a macro with a star and a non-star mode.

The extra level of indirection is necessary to parse as a KaTeX macro.
see:
- https://latexref.xyz/_005c_0040ifstar.html
- https://katex.org/docs/supported.html#macros
"""
function starmacro(name, num, nostar, star)
    lxd(name, 0, """
        \\providecommand{\\$(name)helper}{
          \\@ifstar{\\$(name)star}{\\$(name)nostar}
        }
        \\providecommand{\\$(name)nostar}[$num]{$nostar}
        \\providecommand{\\$(name)star}[$num]{$star}
        \\$(name)helper"""
    )
end

function starmacro(name, num, nostar)
    left = ["(", "[", "\\lvert", "\\lVert", "\\langle"]
    middle = ["|", "\\vert", "\\|", "\\Vert"]
    right = [")", "]", "\\rvert", "\\rVert", "\\rangle"]
    star = replace(
        nostar,
        [
            Pair(delim, direction * delim)
            for (direction, delims) in zip(
                ("\\left", "\\middle", "\\right"),
                (left, middle, right)
            ) for delim in delims
        ]...,
    )
    starmacro(name, num, nostar, star)
end

"""
DeclarePairedDelimiterX(name, num, left, right, body)

An emulator of the corresponding mathtools macro.
"""
function DeclarePairedDelimiterX(name, num, left, right, body)
    starmacro(
        name,
        num,
        "$left $body $right",
        "\\left $left $body \\right $right",
    )
end

"""
DeclarePairedDelimiter(name, left, right)

An emulator of the corresponding mathtools macro.
"""
function DeclarePairedDelimiter(name, left, right)
    DeclarePairedDelimiterX(name, 1, left, right, "#1")
end

# LaTeX-like math macros.
# n.b.: these are parsed by Franklin, *not* KaTeX.
# KaTeX has its own macro system: https://katex.org/docs/supported.html#macros
const katex_commands = [
    # colors
    lxd("silver",        0, raw"""#9e9997"""),
    lxd("lightblue",     0, raw"""#a1b4c7"""),
    lxd("seagreen",      0, raw"""#23553c"""),
    lxd("orange",        0, raw"""#ea8810"""),
    lxd("rust",          0, raw"""#b8420f"""),
    lxd("lightsilver",   0, raw"""#e7e6e5"""),
    lxd("darksilver",    0, raw"""#96918f"""),
    lxd("darklightblue", 0, raw"""#8999a9"""),
    lxd("darkseagreen",  0, raw"""#1e4833"""),
    lxd("darkorange",    0, raw"""#c7740e"""),
    lxd("darkrust",      0, raw"""#9c380d"""),
    # general
    lxd("defeq", 0, raw"""\coloneqq"""),
    lxd("BigO", 0, raw"""\mathcal{O}"""),
    lxd("Id", 0, raw"""\text{Id}"""),
    lxd("vec", 1, raw"""\bm{#1}"""),  # \renewcommmand
    lxd("T", 0, raw"""\top"""),
    lxd("dd", 0, raw"""\, \text{d}"""),
    lxd("Reverse", 0, raw"""\updownarrow"""),
    lxd("Nat", 0, raw"""\mathbb{N}"""),
    lxd("Int", 0, raw"""\mathbb{Z}"""),
    lxd("Integers", 0, raw"""\mathbb{Z}"""),
    lxd("Rationals", 0, raw"""\mathbb{Q}"""),
    lxd("Q", 0, raw"""\mathbb{Q}"""),
    # \Reals and \Complex already provided by KaTeX
    lxd("C", 0, raw"""\mathbb{C}"""),
    # paired delimiters
    DeclarePairedDelimiter("norm", raw"""\lVert""", raw"""\rVert"""),
    DeclarePairedDelimiter("card", raw"""\lvert""", raw"""\rvert"""),
    DeclarePairedDelimiter("abs", raw"""\lvert""", raw"""\rvert"""),
    DeclarePairedDelimiter("floor", raw"""\lfloor""", raw"""\rfloor"""),
    DeclarePairedDelimiter("ceil", raw"""\lceil""", raw"""\rceil"""),
    DeclarePairedDelimiterX(
        "fro",
        1,
        raw"""\lVert""",
        raw"""\rVert_{\operatorname{FRO}}""",
        "#1"
    ),
    DeclarePairedDelimiterX(
        "inner",
        2,
        raw"""\langle""",
        raw"""\rangle""",
        "#1, #2",
    ),
    # operators
    lxd("argmin", 0, raw"""\operatorname*{argmin}"""),
    lxd("argmax", 0, raw"""\operatorname*{argmax}"""),
    lxd("curl", 0, raw"""\operatorname{curl}"""),
    lxd("diag", 0, raw"""\operatorname{diag}"""),
    lxd("trace", 0, raw"""\operatorname{trace}"""),
    lxd("logdet", 0, raw"""\operatorname{logdet}"""),
    lxd("chol", 0, raw"""\operatorname{chol}"""),
    # probability
    lxd("p", 0, raw"""\pi"""),
    lxd("N", 0, raw"""\mathcal{N}"""),
    lxd("mean", 0, raw"""\mean"""),
    lxd("var", 0, raw"""\sigma^2"""),
    lxd("std", 0, raw"""\sigma"""),
    starmacro("E", 1, raw"""\mathbb{E}[#1]"""),
    starmacro("Ep", 2, raw"""\mathbb{E}_{#1}[#2]"""),
    starmacro("Var", 1, raw"""\mathbb{V}\text{ar}[#1]"""),
    starmacro("Cov", 2, raw"""\mathbb{C}\text{ov}[#1, #2]"""),
    starmacro("Corr", 2, raw"""\mathbb{C}\text{orr}[#1, #2]"""),
    starmacro("Entropy", 1, raw"""\mathbb{H}[#1]"""),
    starmacro("MI", 2, raw"""\mathbb{I}[#1; #2]"""),
    starmacro("KL", 2, raw"""\mathbb{D}_{\operatorname{KL}}(#1\; \| \;#2)"""),
]

const katex_environments = append!(
    [
        lxe(name, 0, make_def(name))
        for name in ["definition", "lemma", "theorem", "corollary"]
    ],
    [
        lxe("proof", 0, raw"""
            \begin{wrap}{div class="proof"}
            _Proof_.""" => raw"""
            \begin{wrap}{span class="qedsymbol"}$ \square $\end{wrap}
            \end{wrap}
            """
        ),
    ],
)

for store in (katex_commands, katex_environments), (name, def) in store
    GLOBAL_LXDEFS[name] = def
end

