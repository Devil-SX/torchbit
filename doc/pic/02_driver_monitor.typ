#set page(width: 1360pt, height: 500pt, margin: 60pt, fill: white)
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

#let dut-node(body) = {
  box(
    width: 140pt, height: 90pt,
    fill: rgb("#DBEAFE"),
    stroke: 3pt + rgb("#3B82F6"),
    radius: 14pt,
    align(center + horizon, text(weight: "bold", size: 26pt, body))
  )
}

#let label-text(body) = {
  text(size: 17pt, fill: rgb("#6B7280"), style: "italic", body)
}

#let solid-arrow = text(size: 30pt, fill: rgb("#10B981"), weight: "bold", sym.arrow.r)
#let dash-arrow-purple = text(size: 26pt, fill: rgb("#8B5CF6"), sym.arrow.r.dashed)
#let solid-arrow-orange = text(size: 30pt, fill: rgb("#F97316"), weight: "bold", sym.arrow.r)

#align(center + horizon,
  grid(
    columns: (auto, 72pt, auto, 72pt, auto, 72pt, auto, 72pt, auto),
    align: center + horizon,
    // LogicSequence (input)
    node("LogicSequence", rgb("#F3E8FF"), rgb("#8B5CF6"), width: 180pt),
    // dashed arrow
    stack(dir: ttb, spacing: 8pt,
      label-text("back-door"),
      dash-arrow-purple,
      label-text("load()"),
    ),
    // Driver
    node("Driver", rgb("#D1FAE5"), rgb("#10B981")),
    // solid arrow
    stack(dir: ttb, spacing: 8pt,
      label-text("front-door"),
      solid-arrow,
      label-text("data + valid"),
    ),
    // DUT
    dut-node("DUT"),
    // solid arrow out
    stack(dir: ttb, spacing: 8pt,
      label-text("front-door"),
      solid-arrow-orange,
    ),
    // Monitors column
    stack(dir: ttb, spacing: 24pt,
      node("PoolMonitor", rgb("#FFEDD5"), rgb("#F97316"), width: 200pt),
      node("FIFOMonitor", rgb("#FFEDD5"), rgb("#F97316"), width: 200pt),
    ),
    // dashed arrow out
    stack(dir: ttb, spacing: 8pt,
      label-text("back-door"),
      dash-arrow-purple,
      label-text("dump()"),
    ),
    // LogicSequence (output)
    node("LogicSequence", rgb("#F3E8FF"), rgb("#8B5CF6"), width: 180pt),
  )
)
