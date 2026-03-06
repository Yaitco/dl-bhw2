#import "@preview/quick-maths:0.1.0": shorthands
#import "@preview/cetz-plot:0.1.0": plot
#import "@preview/cetz:0.3.2"

#let my_name = [Маркотенко Александр]

#let project(body, hw: "", date: "", number: "", show-author: true, type: [Домашние задание], author: [Маркотенко Александр]) = {
  set page(
    margin: (top: 1.2in, left: 0.5in, right: 0.5in),
    numbering: "1 / 1",
    header-ascent: 30%,
    header: block(width: 105%, {
    h(-2.5%)
    box({
      grid(
        columns: (1fr, 1fr),
        align(left, {
          text(font: "HSE Slab", weight: "light", 1em, hw)
          linebreak()
          text(font: "HSE Slab", weight: "bold", 1em, [#type #number])
        }),
        align(right, {
          text(font: "HSE Slab", weight: "regular", 1em, if show-author { author } else { "" })
          linebreak()
          text(font: "HSE Slab", weight: "regular", 1em, date)
        })
      )
     line(length: 100%)
    })
  })
  )
  set par(justify: true,)
  set text(font: "HSE Sans", fill: rgb("060606"))
  show math.equation: text.with(features: ("cv01",))
  show: shorthands.with(
    ($+-$, $plus.minus$),
    ($==$, $equiv$),
    ($!==$, $equiv.not$),
    ($\\$, $without$),
    ($<<$, $angle.l$),
    ($>>$, $angle.r$),
    ($~=$, $tilde.equiv$),
    ($++$, $plus.circle$),
    ($k*$, $times.circle$),
    ($*k$, $times.circle$),
    ($*$, $dot.op$),
    ($[[$, $bracket.double.l$),
    ($]]$, $bracket.double.r$),
  )
  // show regex("[,]"): text.with(font: "New Computer Modern")
  show math.arrow.r.twohead: math.arrows.rr
  show math.integral: math.limits
  show math.integral.double: math.limits
  show math.integral.triple: math.limits
  show math.sum: math.limits
  body
}

#let color-box = block.with(
  width: 100%,
  radius: 10pt,
  stroke: 0pt,
  inset: 10pt,
)



#let problem(..args) = {
  show math.equation: it => if repr(it.at("body").func()) != "math-style" {
    math.display(it)
  } else {
    it
  }
  assert(args.named() == (:))

  let args = args.pos()

  assert(1 <= args.len() and args.len() <= 2)

  let body = args.last()
  let number = if args.len() == 2 {
    args.first()
  } else {
    none
  }


  color-box(
    // stroke: (left: (thickness: 2pt)),
    stroke: (
        left: (thickness: 2pt, 
        // cap: "round", dash: (10pt, 0pt), 
        // paint: gradient.linear(black, white, angle: 90deg)
        ), 
        // top: (thickness: 2pt, 
        // cap: "round", dash: (10pt, 0pt), paint: gradient.linear(black, white)
        // )
        ),
    radius: 10pt,
    // fill: blue.lighten(95%),
    { 
      // set text(font: "HSE Slab")
      if number != none [*#number*]
      set text(font: "HSE Sans")
      body
    }
  )
}

#let solution = color-box.with()

#let answer(body) = {
  // show math.equation: it => if repr(it.at("body").func()) != "math-style" {
  //   math.display(it) }
  color-box(
  stroke: 1pt,
  radius: 20pt,
  [Ответ: ]+
  body,
  )
}

//
#let Qed = align(right, $display(qed)$)
#let eps = math.epsilon
#let PP = $bb(P)$
#let dx = $dif x$
#let dy = $dif y$
#let dz = $dif z$
#let dr = $dif r$
#let dh = $dif h$
#let dt = $dif t$
#let dphi = $dif phi$
#let dtheta = $dif theta$
#let Bin = math.op("Bin")
#let Var = math.op([$VV$ar])
#let Cov = math.op([$CC$ov])
#let Corr = math.op([$CC$orr])
#let pCorr = math.op([p$CC$orr])
#let Unif = math.op("Unif")
#let arctg = math.op("arctg")
#let plim = math.op("plim")
#let conj = math.overline
#let vec = math.vec.with(delim: "[")
#let mat = math.mat.with(delim: "[")
#let Vec = math.op("Vec")
#let diag = math.op("diag")
#let rank = math.op("rank")
#let zv = math.ast.basic
//


#let bar = math.scripts(math.class(
  "large",
  pad(right: 2pt, scale(y: 150%, line(length: 15pt, angle: 90deg, stroke: 0.6pt)))
))

#let row(..args) = {
  set align(left)
  table(columns: 1, stroke: none, ..args.named(), ..args.pos().map(math.equation))
}


// #show: project.with()

#let t(it) = box(text(font: "HSE Sans", it))


// #line(angle: 90deg, stroke: (thickness: 2pt, cap: "round"), length: 100% )

// #box(stroke: (left: (thickness: 2pt, cap: "round")), lorem(10))
#let FF = math.cal(math.bold("F"))
$
  &EE(EE(X | FF)) = EE(X) \
  &Var(EE(X | FF)) = Var(EE(X | FF)) + EE(Var(X | FF))
$