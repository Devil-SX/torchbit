#set page(width: 1360pt, height: 480pt, margin: 60pt, fill: white)
#set text(font: "Noto Sans", size: 18pt)

#let node(body, fill-color, stroke-color, width: 180pt, height: 72pt) = {
  box(
    width: width, height: height,
    fill: fill-color,
    stroke: 2.5pt + stroke-color,
    radius: 14pt,
    align(center + horizon, text(weight: "bold", size: 22pt, body))
  )
}

#let arrow-label(body) = {
  text(size: 16pt, fill: rgb("#6B7280"), style: "italic", body)
}

#let domain-box(title, title-color, bg, stroke-color, body) = {
  box(
    fill: bg,
    stroke: 2.5pt + stroke-color,
    radius: 18pt,
    inset: 30pt,
  )[
    #align(center, text(weight: "bold", size: 26pt, fill: title-color, title))
    #v(24pt)
    #body
  ]
}

#align(center + horizon,
  grid(
    columns: (auto, 140pt, auto),
    align: center + horizon,
    // Cocotb domain
    domain-box("Cocotb / HDL", rgb("#4F46E5"), rgb("#EEF2FF"), rgb("#6366F1"),
      align(center,
        grid(
          columns: (auto, 180pt, auto),
          align: center + horizon,
          node("LogicSequence", rgb("#F3E8FF"), rgb("#8B5CF6"), width: 200pt),
          stack(dir: ttb, spacing: 8pt,
            arrow-label("to_logic_sequence " + sym.arrow.r),
            arrow-label(sym.arrow.l + " from_logic_sequence"),
          ),
          node("VectorSequence", rgb("#D1FAE5"), rgb("#10B981"), width: 210pt),
        )
      )
    ),
    // Bridge arrow
    stack(dir: ttb, spacing: 10pt,
      align(center, text(weight: "bold", size: 17pt, fill: rgb("#7C3AED"), "to_matrix\nfrom_matrix")),
      align(center, text(size: 26pt, fill: rgb("#7C3AED"), sym.arrow.l.r)),
    ),
    // PyTorch domain
    domain-box("PyTorch", rgb("#EA580C"), rgb("#FFF7ED"), rgb("#F97316"),
      align(center,
        node("Matrix", rgb("#FEE2E2"), rgb("#EF4444"), width: 200pt)
      )
    ),
  )
)
