#set page(width: 1280pt, height: 780pt, margin: 60pt, fill: white)
#set text(font: "Noto Sans", size: 18pt)

#let node(body, fill-color, stroke-color, width: 180pt, height: 68pt) = {
  box(
    width: width, height: height,
    fill: fill-color,
    stroke: 2.5pt + stroke-color,
    radius: 14pt,
    align(center + horizon, text(weight: "bold", size: 21pt, body))
  )
}

#let label-text(body) = {
  text(size: 17pt, fill: rgb("#6B7280"), style: "italic", body)
}

#let section-title(body, color) = {
  text(weight: "bold", size: 26pt, fill: color, body)
}

#let arrow-right = text(size: 32pt, fill: rgb("#9CA3AF"), sym.arrow.r)

// Front-door row
#align(center,
  box(
    fill: rgb("#F0F9FF"), stroke: 2.5pt + rgb("#3B82F6"), radius: 18pt, inset: 32pt,
  )[
    #align(center, section-title("Front-door  (HDL)", rgb("#2563EB")))
    #v(24pt)
    #align(center,
      grid(
        columns: (auto, 100pt, auto, 100pt, auto),
        align: center + horizon,
        node("DUT", rgb("#DBEAFE"), rgb("#3B82F6"), width: 140pt, height: 80pt),
        stack(dir: ttb, spacing: 8pt, label-text("HDL signals"), arrow-right),
        node("TwoPortBuffer", rgb("#D1FAE5"), rgb("#10B981"), width: 220pt),
        stack(dir: ttb, spacing: 8pt, label-text("read / write"), arrow-right),
        node("Buffer", rgb("#BBF7D0"), rgb("#059669"), width: 160pt, height: 80pt),
      )
    )
  ]
)

#v(36pt)

// Back-door row
#align(center,
  box(
    fill: rgb("#FFFBEB"), stroke: 2.5pt + rgb("#F59E0B"), radius: 18pt, inset: 32pt,
  )[
    #align(center, section-title("Back-door  (Software)", rgb("#D97706")))
    #v(24pt)
    #align(center,
      grid(
        columns: (auto, 100pt, auto, 100pt, auto, 100pt, auto),
        align: center + horizon,
        node("Tensor", rgb("#FEE2E2"), rgb("#EF4444"), width: 140pt),
        stack(dir: ttb, spacing: 8pt, label-text("to_logic_seq()"), arrow-right),
        node("TileMapping", rgb("#FEF3C7"), rgb("#F59E0B"), width: 190pt),
        stack(dir: ttb, spacing: 8pt, label-text("rearrange"), arrow-right),
        node("Matrix", rgb("#E0F2FE"), rgb("#0EA5E9"), width: 160pt),
        stack(dir: ttb, spacing: 8pt, label-text("pack rows"), arrow-right),
        node("LogicSequence", rgb("#F3E8FF"), rgb("#8B5CF6"), width: 200pt),
      )
    )
  ]
)

#v(36pt)

// Merge: LogicSequence + AddressMapping -> Buffer
#align(center,
  grid(
    columns: (auto, 100pt, auto, 100pt, auto),
    align: center + horizon,
    node("LogicSequence", rgb("#F3E8FF"), rgb("#8B5CF6"), width: 200pt),
    stack(dir: ttb, spacing: 8pt, label-text("values"), arrow-right),
    node("Buffer", rgb("#BBF7D0"), rgb("#059669"), width: 160pt, height: 80pt),
    stack(dir: ttb, spacing: 8pt, arrow-right, label-text("addresses")),
    node("AddressMapping", rgb("#F3E8FF"), rgb("#8B5CF6"), width: 220pt),
  )
)
