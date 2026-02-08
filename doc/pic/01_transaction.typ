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
          columns: (auto, 130pt, auto, 130pt, auto),
          align: center + horizon,
          node("BitStruct", rgb("#FEF3C7"), rgb("#F59E0B")),
          stack(dir: ttb, spacing: 8pt,
            arrow-label("to_logic " + sym.arrow.r),
            arrow-label(sym.arrow.l + " from_logic"),
          ),
          node("Logic", rgb("#F3F4F6"), rgb("#6B7280"), width: 120pt),
          stack(dir: ttb, spacing: 8pt,
            arrow-label("to_logic " + sym.arrow.r),
            arrow-label(sym.arrow.l + " from_logic"),
          ),
          node("Vector", rgb("#D1FAE5"), rgb("#10B981")),
        )
      )
    ),
    // Bridge arrow
    stack(dir: ttb, spacing: 10pt,
      align(center, text(weight: "bold", size: 17pt, fill: rgb("#7C3AED"), "to_array\nfrom_array")),
      align(center, text(size: 26pt, fill: rgb("#7C3AED"), sym.arrow.l.r)),
    ),
    // PyTorch domain
    domain-box("PyTorch", rgb("#EA580C"), rgb("#FFF7ED"), rgb("#F97316"),
      align(center,
        node("Array", rgb("#FEE2E2"), rgb("#EF4444"), width: 200pt)
      )
    ),
  )
)
