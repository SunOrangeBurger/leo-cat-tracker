const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageBreak, LevelFormat, Header, Footer, PageNumber, VerticalAlign,
} = require("docx");

const FONT = "Times New Roman";
const SZ = 24; // 12pt in half-points
const SZ_SMALL = 22;
const CM2 = 1134; // 2cm margins in twips
const A4 = { width: 11906, height: 16838 };
const LINE15 = { line: 360, lineRule: "auto" }; // 1.5 line spacing
const PLACEHOLDER = "BFBFBF";

// ---------- helpers ----------
function ph(text) { // placeholder run - italic, bracketed
  return new TextRun({ text: `<${text}>`, italics: true, font: FONT, size: SZ, color: "C00000" });
}
function run(text, opts = {}) { return new TextRun({ text, font: FONT, size: SZ, ...opts }); }

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    alignment: AlignmentType.LEFT,
    spacing: { before: 360, after: 220, ...LINE15 },
    children: [new TextRun({ text, font: FONT, size: SZ, bold: true, underline: {} })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    alignment: AlignmentType.LEFT,
    spacing: { before: 260, after: 160, ...LINE15 },
    children: [new TextRun({ text, font: FONT, size: SZ, bold: true, underline: {} })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    alignment: AlignmentType.LEFT,
    spacing: { before: 200, after: 120, ...LINE15 },
    children: [new TextRun({ text, font: FONT, size: SZ, bold: true })],
  });
}
function p(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 200, ...LINE15 },
    children: [run(text, opts)],
  });
}
function pRuns(children, opts = {}) {
  return new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200, ...LINE15 }, ...opts, children });
}
function centerP(text, opts = {}) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160, ...LINE15 }, children: [run(text, opts)] });
}
function bullet(text, level = 0) {
  return new Paragraph({
    bullet: { level },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 120, ...LINE15 },
    children: [run(text)],
  });
}
function boldBullet(label, rest, level = 0) {
  return new Paragraph({
    bullet: { level },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 120, ...LINE15 },
    children: [run(label, { bold: true }), run(rest)],
  });
}
function cell(text, { bold = false, shade = null, width, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({ alignment: align, spacing: LINE15, children: [run(text, { bold })] })],
  });
}
function tocLine(numberText, titleText, page) {
  return new Paragraph({
    spacing: { after: 100, ...LINE15 },
    tabStops: [{ type: "right", position: 9500, leader: "dot" }],
    children: [
      run(`${numberText}  ${titleText}`),
      new TextRun({ text: `\t${page}`, font: FONT, size: SZ }),
    ],
  });
}

// page setup shared by chapter sections
function pageProps(titlePage) {
  return {
    page: { size: A4, margin: { top: CM2, bottom: CM2, left: CM2, right: CM2 } },
    titlePage,
  };
}
function chapterHeader(text) {
  return new Header({ children: [ new Paragraph({
    alignment: AlignmentType.RIGHT,
    border: { bottom: { color: "999999", space: 4, style: BorderStyle.SINGLE, size: 4 } },
    children: [run(text, { italics: true, size: SZ_SMALL, color: "595959" })],
  })]});
}
function blankHeader() { return new Header({ children: [new Paragraph({ children: [] })] }); }
function pageFooter() {
  return new Footer({ children: [ new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [ run("", { size: SZ_SMALL }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SZ_SMALL }) ],
  })]});
}

function chapterSection(headerText, children) {
  return {
    properties: pageProps(true),
    headers: { default: chapterHeader(headerText), first: blankHeader() },
    footers: { default: pageFooter(), first: pageFooter() },
    children,
  };
}

// =====================================================================
// FRONT MATTER (no header, no footer, no page numbers)
// =====================================================================
const frontMatter = {
  properties: pageProps(false),
  children: [
    // ---------------- TITLE PAGE ----------------
    centerP("PES UNIVERSITY", { bold: true, size: 28 }),
    centerP("(Established under Karnataka Act No. 16 of 2013)", { size: SZ_SMALL }),
    new Paragraph({ spacing: { before: 900 }, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Internship at PES Research Foundation", italics: true, bold: true, font: FONT, size: 30 })] }),
    new Paragraph({ spacing: { before: 500 }, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "\u201cAI-Enabled Data Association for Space Object Tracking\u201d", bold: true, font: FONT, size: 30 })] }),
    new Paragraph({ spacing: { before: 200 }, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "(ISRO Problem Statement #15 \u2014 Space Situational Awareness)", italics: true, font: FONT, size: SZ })] }),

    new Paragraph({ spacing: { before: 700 }, alignment: AlignmentType.CENTER, children: [run("Submitted by:", { italics: true })] }),
    new Table({
      alignment: AlignmentType.CENTER,
      width: { size: 6800, type: WidthType.DXA },
      columnWidths: [3800, 3000],
      rows: [
        new TableRow({ children: [cell("Name", { bold: true, shade: "EDEDED", width: 3800, align: AlignmentType.CENTER }), cell("SRN", { bold: true, shade: "EDEDED", width: 3000, align: AlignmentType.CENTER })] }),
        new TableRow({ children: [cell("Adil Ahmed", { width: 3800, align: AlignmentType.CENTER }), new TableCell({ width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [ph("Enter SRN")] })] })] }),
        new TableRow({ children: [cell("Arun Hariharan", { width: 3800, align: AlignmentType.CENTER }), new TableCell({ width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [ph("Enter SRN")] })] })] }),
        new TableRow({ children: [cell("Ganavi Purushothama", { width: 3800, align: AlignmentType.CENTER }), new TableCell({ width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [ph("Enter SRN")] })] })] }),
        new TableRow({ children: [cell("Pratik Patil", { width: 3800, align: AlignmentType.CENTER }), new TableCell({ width: { size: 3000, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [ph("Enter SRN")] })] })] }),
      ],
    }),

    new Paragraph({ spacing: { before: 700 }, alignment: AlignmentType.CENTER, children: [run("Under the guidance of", { italics: true })] }),
    new Table({
      alignment: AlignmentType.CENTER,
      width: { size: 5600, type: WidthType.DXA },
      columnWidths: [5600],
      borders: {
        top: { style: BorderStyle.SINGLE, size: 6, color: "000000" }, bottom: { style: BorderStyle.SINGLE, size: 6, color: "000000" },
        left: { style: BorderStyle.SINGLE, size: 6, color: "000000" }, right: { style: BorderStyle.SINGLE, size: 6, color: "000000" },
      },
      rows: [ new TableRow({ children: [ new TableCell({
        width: { size: 5600, type: WidthType.DXA }, margins: { top: 160, bottom: 160, left: 160, right: 160 },
        children: [
          new Paragraph({ alignment: AlignmentType.CENTER, children: [run("Vandana M L", { bold: true })] }),
          new Paragraph({ alignment: AlignmentType.CENTER, children: [ph("Designation"), run(", PES Research Foundation")] }),
        ],
      })]})],
    }),

    new Paragraph({ spacing: { before: 700 }, alignment: AlignmentType.CENTER, children: [run("June 2026 \u2013 August 2026", { bold: true })] }),

    new Paragraph({ spacing: { before: 900 }, alignment: AlignmentType.CENTER, children: [run("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", { bold: true })] }),
    centerP("FACULTY OF ENGINEERING"),
    centerP("PES UNIVERSITY", { bold: true }),
    centerP("100ft Ring Road, Bengaluru \u2013 560 085, Karnataka, India", { size: SZ_SMALL }),

    new Paragraph({ children: [new PageBreak()] }),

    // ---------------- CERTIFICATE PLACEHOLDER ----------------
    new Paragraph({ spacing: { before: 3000 }, alignment: AlignmentType.CENTER, children: [run("ATTACH COMPANY / MENTOR CERTIFICATE HERE", { bold: true, size: 28 })] }),
    centerP("(In a format as per PES Research Foundation policy.)", { italics: true }),
    centerP("If unavailable at submission time, attach the first page of the internship acceptance / joining letter instead.", { italics: true }),
    new Paragraph({ children: [new PageBreak()] }),

    // ---------------- DECLARATION ----------------
    centerP("DECLARATION", { bold: true, underline: {}, size: 28 }),
    new Paragraph({ spacing: { before: 300, after: 240, ...LINE15 }, alignment: AlignmentType.JUSTIFIED, children: [
      run("We hereby declare that the project entitled "),
      run("\u201cAI-Enabled Data Association for Space Object Tracking\u201d", { bold: true }),
      run(" has been carried out at PES Research Foundation by us, under the guidance of Vandana M L, "),
      ph("Designation"),
      run(", PES Research Foundation, and submitted in partial fulfilment of the credits for the degree of "),
      run("Bachelor of Technology", { bold: true }),
      run(" in "),
      run("Computer Science and Engineering", { bold: true }),
      run(" of "),
      run("PES University, Bengaluru", { bold: true }),
      run(" during the academic period June 2026 \u2013 August 2026. The matter embodied in this report has not been submitted to any other university or institution for the award of any degree."),
    ]}),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [2800, 3760, 2800],
      rows: [
        new TableRow({ children: [cell("SRN", { bold: true, shade: "EDEDED", width: 2800, align: AlignmentType.CENTER }), cell("Name", { bold: true, shade: "EDEDED", width: 3760, align: AlignmentType.CENTER }), cell("Signature", { bold: true, shade: "EDEDED", width: 2800, align: AlignmentType.CENTER })] }),
        ...["Adil Ahmed", "Arun Hariharan", "Ganavi Purushothama", "Pratik Patil"].map(name => new TableRow({ children: [
          new TableCell({ width: { size: 2800, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [ph("SRN")] })] }),
          cell(name, { width: 3760, align: AlignmentType.CENTER }),
          new TableCell({ width: { size: 2800, type: WidthType.DXA }, children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [ph("Signature")] })] }),
        ]})),
      ],
    }),
    new Paragraph({ children: [new PageBreak()] }),

    // ---------------- ACKNOWLEDGEMENT ----------------
    centerP("ACKNOWLEDGEMENT", { bold: true, underline: {}, size: 28 }),
    p("We would like to express our sincere gratitude to our guide, Vandana M L, PES Research Foundation, for her continuous guidance, assistance, and encouragement throughout the course of this internship."),
    p("We are grateful to the internship coordinators, Dr. Animesh Giri and Dr. Prema R, Department of Computer Science and Engineering, PES University \u2013 Electronic City Campus, for organizing, managing, and helping out with the entire process."),
    pRuns([ run("We are grateful to the Capstone Project guide, "), ph("Capstone Project Guide Name"), run(", Department of Computer Science and Engineering, PES University \u2013 Electronic City Campus, for organizing, managing, and helping out with the entire process.") ]),
    p("We take this opportunity to thank Dr. Sandesh B J, Chairperson, Department of Computer Science and Engineering, PES University \u2013 Electronic City Campus, for all the knowledge and support we have received from the department."),
    p("We would like to thank Dr. B. K. Keshavan, Dean of Faculty, PES University, for his help."),
    p("We are deeply grateful to Prof. Jawahar Doreswamy, Chancellor, PES University, Dr. Suryaprasad J, Vice-Chancellor, PES University, and Prof. Nagarjuna Sadineni, Pro-Vice-Chancellor, PES University, for providing us various opportunities and enlightenment every step of the way."),
    p("Finally, this internship could not have been completed without the continual support and encouragement we have received from our parents and friends."),
    new Paragraph({ children: [new PageBreak()] }),

    // ---------------- ABSTRACT ----------------
    centerP("ABSTRACT", { bold: true, underline: {}, size: 28 }),
    p("Low Earth Orbit is increasingly crowded with active satellites, fragmentation debris, and natural micrometeoroids, all producing similar raw sensor observations. The classical answer to associating a new observation with a catalogued object, Joint Probabilistic Data Association (JPDA), is well understood but degrades combinatorially as object density increases \u2014 precisely the regime that matters most as large constellations grow. This internship, carried out under PES Research Foundation against ISRO Problem Statement #15, set out to test whether a learned, neighbourhood-aware association system \u2014 vector embeddings combined with a graph neural network (GNN) reasoning only over small local neighbourhoods \u2014 could match or beat JPDA specifically in dense, closely-spaced-object regimes."),
    p("The project moved through three architectural phases over approximately eight weeks: an initial three-module proposal (a GNN association engine, a new-object track-initiation module, and a closed-loop reinforcement-learning sensor-tasking agent); a mid-project pivot to an MLP object-class classifier feeding three independent, class-specific GNNs; and a final validation sprint that consolidated the design into a single, class-agnostic two-stage pipeline \u2014 physics-based orbital propagation, approximate-nearest-neighbour retrieval, and a GNN invoked only on genuinely ambiguous cases. This final pipeline was evaluated end-to-end on a 20,000-object synthetic catalog, achieving 100% correct matches across 56,841 detections, 99.9% catalog completeness, and 100% candidate retrieval recall, with 82.4% of detections resolved by the fast retrieval stage alone. Two genuine failure modes \u2014 a circular validation diagnostic unable to detect a live/train mismatch, and silent overfitting caused by an undersized validation split \u2014 were discovered and fixed during the sprint, and are reported in this document as the project's most concrete evidence of research rigour."),
    p("The report also states candidly what remains open: the head-to-head benchmark against JPDA that the project's own success criteria depend on has not yet been run, the two additional proposed modules (track initiation and sensor tasking) were not built, and the validated Python model has not yet been reconnected to the pre-existing, higher-performance Rust simulator. These are identified as the clear next steps arising from this internship."),
    new Paragraph({ children: [new PageBreak()] }),

    // ---------------- TABLE OF CONTENTS ----------------
    centerP("TABLE OF CONTENTS", { bold: true, underline: {}, size: 28 }),
    new Paragraph({ spacing: { before: 200 } }),
    tocLine("1.", "Introduction", "1"),
    tocLine("1.1", "Organisation Brief", "1"),
    tocLine("1.2", "Internship Programme Context", "1"),
    tocLine("1.3", "Team, Department, and Role within the Organisation", "2"),
    tocLine("2.", "Internship Project Details", "3"),
    tocLine("2.1", "Project Title", "3"),
    tocLine("2.2", "Relationship to a Larger Programme", "3"),
    tocLine("2.3", "Roles and Responsibilities", "3"),
    tocLine("2.4", "Project Abstract and Scope", "4"),
    tocLine("2.5", "Project Design Details and Technologies Used", "5"),
    tocLine("2.6", "Implementation Details", "7"),
    tocLine("2.7", "Project Results, Accomplishments, and Learning Outcomes", "9"),
    tocLine("3.", "Conclusion", "12"),
    tocLine("4.", "References / Bibliography", "13"),
    tocLine("", "Appendices", "14"),
    new Paragraph({ children: [new PageBreak()] }),

    // ---------------- LIST OF TABLES AND FIGURES ----------------
    centerP("LIST OF TABLES AND FIGURES", { bold: true, underline: {}, size: 28 }),
    new Paragraph({ spacing: { before: 200 } }),
    tocLine("Table 2.1", "Module ownership across the team", "4"),
    tocLine("Table 2.2", "MLP object-class classifier \u2014 configuration and results (Phase 2)", "6"),
    tocLine("Table 2.3", "leo-cat-tracker repository layout", "8"),
    tocLine("Table 2.4", "Final pipeline \u2014 headline evaluation results", "10"),
    tocLine("Table 2.5", "Progress against the original 12-week plan", "11"),
    tocLine("Table A.1", "rust-leo-sim repository layout", "14"),
    new Paragraph({ children: [new PageBreak()] }),
  ],
};

// =====================================================================
// CHAPTER 1 — INTRODUCTION
// =====================================================================
const chapter1 = chapterSection("Chapter 1 \u2014 Introduction", [
  h1("1. Introduction"),

  h2("1.1 Organisation Brief"),
  p("PES Research Foundation is the research arm of PES University, Bengaluru, and runs the PESURF (PES University Research Foundation) programme through which undergraduate students undertake mentored, faculty-supervised research internships on externally posed problem statements. This internship was carried out under PESURF against ISRO Problem Statement #15, issued in the domain of Space Situational Awareness (SSA), with the objective of exploring learned, AI-based alternatives to classical data-association methods used in satellite catalog maintenance."),
  p("The programme is structured as a mentored research track rather than a conventional corporate internship: teams work under a faculty research mentor, are expected to produce a literature-grounded proposal, present progress at a formal mid-project evaluation before a faculty panel, and deliver a final, validated piece of research work by the end of the internship period."),

  h2("1.2 Internship Programme Context"),
  p("The internship ran for approximately eight weeks, from June 2026 to August 2026, structured around three milestones: a literature review and research proposal (June 2026), a mid-project faculty evaluation (July 2026), and a final validation sprint concluding in this report (August 2026). Eligibility and team formation were handled by the PESURF coordinators, Dr. Animesh Giri and Dr. Prema R, Department of Computer Science and Engineering, PES University \u2013 Electronic City Campus."),
  p("The programme's stated objective for this problem statement was to determine whether a learned, graph-neural-network-based approach to space-object data association could outperform the classical Joint Probabilistic Data Association (JPDA) baseline in dense, closely-spaced-object regimes \u2014 the regime in which JPDA is known to degrade combinatorially and which matters increasingly as satellite constellations grow larger and denser."),

  h2("1.3 Team, Department, and Role within the Organisation"),
  p("The team consisted of four undergraduate students from the Department of Computer Science and Engineering \u2014 Adil Ahmed, Arun Hariharan, Ganavi Purushothama, and Pratik Patil \u2014 working under the direct guidance of research mentor Vandana M L at PES Research Foundation. The team operated as a self-contained research unit: responsible for its own literature review, system design, implementation, evaluation, and reporting, with the mentor providing direction, review, and go/no-go feedback at each milestone rather than day-to-day task assignment."),
  p("Within the team, work was divided by module so that pieces could be developed and validated independently before being merged \u2014 detailed in Section 2.3 \u2014 with all four members jointly responsible for the final validated pipeline described in Section 2.5 onward."),
]);

// =====================================================================
// CHAPTER 2 — INTERNSHIP PROJECT DETAILS
// =====================================================================
const chapter2 = chapterSection("Chapter 2 \u2014 Internship Project Details", [
  h1("2. Internship Project Details"),

  h2("2.1 Project Title"),
  p("AI-Enabled Data Association for Space Object Tracking, addressing ISRO Problem Statement #15 (Space Situational Awareness)."),

  h2("2.2 Relationship to a Larger Programme"),
  p("This project is not a standalone academic exercise: it was posed as ISRO Problem Statement #15 and framed explicitly against ISRO's NETRA programme, which performs ground-based tracking of Low Earth Orbit objects down to roughly 10 cm in size. ISRO currently maintains awareness of over 27,000 tracked objects, with an estimated 80 dangerous close approaches occurring per week across the catalog. The project's data-association engine is designed as a component that could, in principle, sit inside a NETRA-style catalog-maintenance pipeline \u2014 it is a contribution toward a larger, real operational problem rather than a self-contained toy system."),

  h2("2.3 Roles and Responsibilities"),
  p("As presented at the mid-project evaluation, work was split by module so that each piece could be developed and validated independently before being merged into a unified pipeline:"),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2400, 3000, 3960],
    rows: [
      new TableRow({ children: [cell("Member", { bold: true, shade: "EDEDED", width: 2400 }), cell("Module Owned", { bold: true, shade: "EDEDED", width: 3000 }), cell("Notes", { bold: true, shade: "EDEDED", width: 3960 })] }),
      new TableRow({ children: [cell("Pratik Patil", { width: 2400 }), cell("MLP object-class classifier", { width: 3000 }), cell("Completed and validated as scoped (Section 2.7)", { width: 3960 })] }),
      new TableRow({ children: [cell("Arun Hariharan", { width: 2400 }), cell("Satellite-class GNN", { width: 3000 }), cell("Co-owner of the Rust orbital simulator", { width: 3960 })] }),
      new TableRow({ children: [cell("Adil Ahmed", { width: 2400 }), cell("Man-made debris GNN", { width: 3000 }), cell("Co-owner of the Rust orbital simulator", { width: 3960 })] }),
      new TableRow({ children: [cell("Ganavi Purushothama", { width: 2400 }), cell("Natural debris GNN", { width: 3000 }), cell("\u2014", { width: 3960 })] }),
    ],
  }),
  pRuns([run("Table 2.1: Module ownership across the team.", { italics: true, size: SZ_SMALL })], { alignment: AlignmentType.CENTER, spacing: { before: 100, after: 300 } }),
  p("All four members jointly own the final, consolidated pipeline described from Section 2.5 onward, since the class-specific GNN split above was folded into a single class-agnostic model during the final validation sprint (explained in Section 2.5). Collaboration was structured around weekly syncs with the mentor and a shared codebase, with the two Rust-simulator co-owners also responsible for keeping the classical baseline available for the eventual JPDA comparison discussed in Section 3."),

  h2("2.4 Project Abstract and Scope"),
  p("Two linked problems must be solved continuously for every object in Low Earth Orbit: identification (given a new sensor measurement, is this a previously catalogued object, and which one?) and cataloguing (if not a confident match, when should a new track be initiated, and how should the catalog update?). The classical answer to identification, JPDA, computes an association probability for essentially every plausible observation-object pairing, which becomes computationally intractable as hundreds of objects occupy closely spaced, near-identical orbital shells \u2014 precisely the regime that matters most as large constellations grow."),
  p("The project's falsifiable hypothesis was that a learned, neighbourhood-aware association system \u2014 vector embeddings combined with a GNN reasoning only over small local neighbourhoods, rather than the full joint hypothesis space JPDA evaluates \u2014 could match or beat JPDA specifically in dense, closely-spaced-object regimes, by learning regime-specific drift and noise patterns that a fixed, hand-tuned probabilistic model does not capture. Three outcomes were pre-agreed before any testing began, specifically so the result could not be argued into whatever was convenient afterward: a green light (a real margin over JPDA in the dense regime, with high retrieval recall), a yellow light (roughly tied, or winning only on easy cases), or a red light (poor retrieval recall alone, meaning the problem lies upstream of the GNN entirely)."),
  p("The scope of the work delivered within the internship period covers the retrieval-and-cataloguing foundation of this hypothesis in full, evaluated end-to-end on synthetic data (Section 2.7), together with a working classical baseline (the Rust simulator, Section 2.5) capable of running the JPDA comparison. It does not yet cover the completed head-to-head JPDA benchmark itself, nor the two additional modules proposed at the outset (new-object track initiation and closed-loop sensor tasking) \u2014 both are identified as clear next steps in Section 3."),

  h2("2.5 Project Design Details and Technologies Used"),
  h3("2.5.1 Literature Grounding"),
  p("The design was grounded in a structured, 13-reference literature review completed in the first two weeks. Classical Kalman-filter-family methods (KF, EKF, UKF) and JPDA were reviewed as the established baseline, with JPDA's combinatorial cost growth, inability to learn from data, and open-loop sensor tasking identified as its core limitations. Four adjacent machine-learning approaches (orbit-prediction error correction, CNN-based debris classification, a YOLOv3-plus-UKF pipeline, and random-finite-set multi-sensor fusion) were reviewed and found to each solve only an isolated sub-problem. Three graph-neural-network papers on multi-object tracking (Li, Gao & Jiang, WACV 2020; Weng et al., CVPR 2020; Wang, Kitani & Weng, ICRA 2021) motivated the core architectural choice: treating data association as bipartite graph matching, with a GNN learning edge weights end-to-end. Full references are listed in Chapter 4."),

  h3("2.5.2 Architectural Evolution"),
  p("The system that was ultimately built and evaluated does not match the original proposal one-for-one, and this report records that evolution explicitly rather than presenting only the final state:"),
  boldBullet("Phase 1 \u2014 Proposal: ", "a three-module system \u2014 a GNN association engine (Module A), a new-object track-initiation module using an admissible-region Gaussian Mixture Model (Module B), and a PPO-based reinforcement-learning sensor-tasking agent closing the loop between association and sensor scheduling (Module C) \u2014 built on top of a pre-existing Rust simulator, scoped to a 12-week plan."),
  boldBullet("Phase 2 \u2014 Mid-project pivot: ", "an MLP object-class classifier (satellite / man-made debris / natural debris) feeding three independent, class-specific GNNs, on the reasoning that different object classes have distinct motion signatures. The classifier achieved 89.8% accuracy with RCS (radar cross-section) features included, reported at the mid-project evaluation as 53% complete (9 of 17 workstreams)."),
  boldBullet("Phase 3 \u2014 Final validation sprint: ", "a single, class-agnostic two-stage pipeline. The pre-existing Rust simulator had two load-bearing problems \u2014 running with SGP4 (real orbital drift) enabled degraded performance badly enough that testing was often done with it disabled, silently removing the exact ambiguity that makes the dense regime hard; and ground truth was semi-hardcoded into the visualisation layer rather than existing as an independent stream. Rather than spend the remaining internship time fixing a simulator whose core hypothesis had not yet been validated, the team built a deliberately throwaway-quality, fast-to-iterate Python validation sprint to cheaply test the idea first."),
  p("The class-specific GNN split from Phase 2 was folded into one class-agnostic model in Phase 3 because the two-stage retrieval-then-GNN architecture generalises across object classes by construction \u2014 the retrieval gate and the local-neighbourhood GNN reason over feature similarity, not object-type identity \u2014 so a per-class split was no longer architecturally necessary once retrieval, rather than classification, became the primary filter. This pivot is treated in this report as a deliberate, reasoned response to two concrete engineering problems discovered mid-project, not as scope drift; the trade-off is that the Phase 2 classifier and class-specific GNNs are not part of the pipeline evaluated in Section 2.7."),

  h3("2.5.3 Final Pipeline Architecture"),
  p("The evaluated system is deliberately split into three stages of increasing computational cost; collapsing them into a single model was considered and rejected during design."),
  boldBullet("Stage 0 \u2014 Physics, not learned: ", "every object's orbital state (five Keplerian elements plus derived features) is propagated analytically using J2 oblateness and atmospheric drag, and every measurement is propagated to a common epoch before any distance comparison \u2014 otherwise the system would conflate a different object with the same object having naturally precessed."),
  boldBullet("Stage 1 \u2014 Vector retrieval, runs on every measurement: ", "each object becomes a normalised feature vector (angles handled via sine/cosine to respect wraparound; size/RCS down-weighted as the noisiest signal) and is looked up via an approximate-nearest-neighbour (ANN) index. Most measurements resolve here alone."),
  boldBullet("Stage 2 \u2014 GNN second opinion, ambiguity only: ", "invoked only when Stage 1's retrieval gate rejects a candidate that is still plausibly close, operating on a small local neighbourhood around that candidate rather than the full catalog graph \u2014 the specific architectural reason the design should scale better than JPDA as density increases, since per-decision cost does not grow with catalog size."),
  p("Technologies used: Python (PyTorch for the GNN and MLP, scikit-learn-adjacent tooling for preprocessing) for the learned pipeline; Rust (with Rayon for parallelism, the sgp4 crate, and egui/eframe for the GUI) for the classical baseline simulator; and TLE-based orbital data conventions (Keplerian elements, SGP4/SDP4 propagation) throughout."),

  h3("2.5.4 Classical Baseline \u2014 the Rust Simulator"),
  p("The pre-existing Rust simulator (repository: rust-leo-sim) models the full classical SSA pipeline \u2014 orbital propagation, ground-based detection, data association, track maintenance, collision detection, and catalog generation \u2014 and remains the intended home for the head-to-head JPDA benchmark. It simulates 100,000+ objects with Rayon-parallelised propagation (roughly 40\u201385 ms total per step at that scale), implements both a nearest-neighbour tracker and a full JPDA tracker, reports Precision, Recall, F1, and OSPA metrics, and can ingest live CelesTrak TLE data for 131+ real satellites in addition to synthetic-object generation. Its documented limitations are circular orbits only in the simple propagation mode, no atmospheric effects outside SGP4, a simplified Gaussian sensor-noise model, and linear (not full nonlinear) closest-approach estimation for collision detection."),

  h2("2.6 Implementation Details"),
  p("The Phase 3 Python pipeline (repository: leo-cat-tracker) is organised into the following modules:"),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3400, 5960],
    rows: [
      new TableRow({ children: [cell("Module", { bold: true, shade: "EDEDED", width: 3400 }), cell("Function", { bold: true, shade: "EDEDED", width: 5960 })] }),
      new TableRow({ children: [cell("data_generator.py", { width: 3400 }), cell("Synthetic Keplerian catalog generation", { width: 5960 })] }),
      new TableRow({ children: [cell("orbital_physics.py", { width: 3400 }), cell("J2 / drag propagation (Stage 0)", { width: 5960 })] }),
      new TableRow({ children: [cell("identity_embedding.py", { width: 3400 }), cell("Feature vectors and normalisation", { width: 5960 })] }),
      new TableRow({ children: [cell("leo_ann_index.py", { width: 3400 }), cell("ANN retrieval index (Stage 1)", { width: 5960 })] }),
      new TableRow({ children: [cell("catalog_gnn.py", { width: 3400 }), cell("GNN second-opinion classifier (Stage 2) and trainer internals", { width: 5960 })] }),
      new TableRow({ children: [cell("train_catalog_gnn.py", { width: 3400 }), cell("Offline trainer entry point", { width: 5960 })] }),
      new TableRow({ children: [cell("ground_station_tracking_simulation.py", { width: 3400 }), cell("The live matching loop", { width: 5960 })] }),
      new TableRow({ children: [cell("compare_live_vs_training.py", { width: 3400 }), cell("Live-vs-training distribution diagnostic (Section 2.7)", { width: 5960 })] }),
      new TableRow({ children: [cell("check_catalog_against_embeddings.py", { width: 3400 }), cell("Catalog/embedding-store cross-check", { width: 5960 })] }),
      new TableRow({ children: [cell("run_full_evaluation.py", { width: 3400 }), cell("Orchestrates the full pipeline end-to-end and renders all evaluation plots", { width: 5960 })] }),
    ],
  }),
  pRuns([run("Table 2.3: leo-cat-tracker repository layout.", { italics: true, size: SZ_SMALL })], { alignment: AlignmentType.CENTER, spacing: { before: 100, after: 300 } }),
  p("A handful of failure modes were designed against before any code was written: timestamp treated as a propagation trigger rather than a raw embedding axis; circular-aware angle encoding; drag parameters sampled rather than hardcoded, since solar activity changes drag-driven decay by an order of magnitude between solar minimum and maximum; and a hard constraint that the GNN never perform full-catalog message passing, since that would erase the entire efficiency argument for the design."),
  p("Because none of the individual pipeline scripts expose structured results on their own \u2014 each prints only to stdout, plus one JSON artefact from the tracking simulation \u2014 an evaluation harness (run_full_evaluation.py) runs the full pipeline end to end (catalog generation \u2192 embedding \u2192 GNN training \u2192 a 60-simulated-second tracking run \u2192 cross-checks), parses each stage's console output against regular expressions matched to that script's own print statements, and renders a fixed set of diagnostic plots. Where the pipeline's dependency graph genuinely allows it, independent stages are run concurrently rather than strictly in sequence; the rest of the pipeline is a true sequential chain that cannot be parallelised without reading a file before it exists."),

  h2("2.7 Project Results, Accomplishments, and Learning Outcomes"),
  h3("2.7.1 Headline Results"),
  p("The full-scale evaluation run used a 20,000-object synthetic catalog, evaluated end-to-end."),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [5760, 3600],
    rows: [
      new TableRow({ children: [cell("Metric", { bold: true, shade: "1F3864", width: 5760 }), cell("Result", { bold: true, shade: "1F3864", width: 3600 })] }),
      new TableRow({ children: [cell("Correct matches / mismatches", { width: 5760 }), cell("100% correct, 0% mismatches (56,841 detections)", { width: 3600 })] }),
      new TableRow({ children: [cell("Catalog completeness", { width: 5760 }), cell("99.9%", { width: 3600 })] }),
      new TableRow({ children: [cell("ANN candidate recall", { width: 5760 }), cell("100%", { width: 3600 })] }),
      new TableRow({ children: [cell("Resolved by fast gate alone", { width: 5760 }), cell("82.4% of detections", { width: 3600 })] }),
      new TableRow({ children: [cell("Genuinely new registrations", { width: 5760 }), cell("17.6% of detections", { width: 3600 })] }),
      new TableRow({ children: [cell("Routed to GNN (ambiguous cases)", { width: 5760 }), cell("0.1% of detections", { width: 3600 })] }),
      new TableRow({ children: [cell("GNN overturn precision (live)", { width: 5760 }), cell("0 / 6 in this run \u2014 sample too small to conclude anything (see 2.7.2)", { width: 3600 })] }),
      new TableRow({ children: [cell("Trained-model val_recall(known), post-fix", { width: 5760 }), cell("\u2248 90%, stable through 250 epochs", { width: 3600 })] }),
    ],
  }),
  pRuns([run("Table 2.4: Final pipeline \u2014 headline evaluation results.", { italics: true, size: SZ_SMALL })], { alignment: AlignmentType.CENTER, spacing: { before: 100, after: 300 } }),
  p("The 0/6 GNN overturn precision figure is reported here exactly as found, rather than smoothed over: every time the GNN overturned a gate rejection to call an object KNOWN in this run, it was wrong. This is a direct structural consequence of how rare a genuine known-in-pool gate failure is by design, and six live events is nowhere near enough sample size to draw a conclusion either way \u2014 it is flagged as an open item in Chapter 3."),

  h3("2.7.2 Accomplishments: Two Failure Modes Found and Fixed"),
  p("The most concrete evidence of research rigour produced during the internship was not a single accuracy number but two genuine bugs, found through deliberate diagnostic work and fixed with evidence."),
  p("First, every early diagnostic \u2014 including the original diagnose_gnn.py \u2014 evaluated the trained model against freshly generated synthetic examples from the same function that produced the training data, which is circular by construction and cannot detect a live/train mismatch. The fix, compare_live_vs_training.py, logged every real GNN invocation during a live tracking run and compared that distribution against a fresh training-style batch. It found that live match distances and margins were consistently smaller than training's, traced to catalog growth: the synthetic catalog grew from roughly 10,000 to over 19,000 objects during a run, producing a denser candidate pool than the old, fixed-size training generator could ever simulate. The fix was not a patch but a redesign: the current harvester, harvest_live_loop_examples(), replays a lightweight version of the actual tracking loop and harvests a training example at every point the live system would genuinely have invoked the GNN, so training data is now a faithful replay of deployment conditions."),
  p("Second, once the harvester was fixed, a full pipeline run showed textbook overfitting \u2014 validation loss bottoming out around epoch 100 of a 250-epoch run and then climbing, while validation recall on known objects degraded and jumped between exact sixth-fractions (1/6, 2/6, 3/6), the tell that revealed the validation split held only about six known-in-pool examples in total. The fix made two changes permanent defaults: the trainer now tracks and restores the best-validation-loss checkpoint rather than saving whatever weights exist at the final epoch, and the minimum-known-examples floor was raised from an implicit 20 to 400. After both fixes, validation recall on known objects climbed to roughly 90% and held stable through 250 epochs."),

  h3("2.7.3 Evaluation Against the Pre-Agreed Success Criteria"),
  p("Measured against the three outcomes pre-agreed in Section 2.4: the retrieval-and-cataloguing half of the hypothesis is strongly validated (100% candidate recall, 99.9% completeness), but the specific claim the project was designed to be judged on \u2014 the GNN's value-add over JPDA on genuinely hard, ambiguous cases \u2014 has not yet been measured, because the head-to-head JPDA comparison has not been run and the live GNN-overturn sample (six events) is too small to be statistically meaningful. The honest position is that the project has a strong foundation but has not yet reached a green, yellow, or red verdict on its own pre-committed question."),

  h3("2.7.4 Learning Outcomes"),
  p("The internship produced learning outcomes across several areas: applied graph neural network design for structured matching problems; orbital mechanics fundamentals (Keplerian elements, J2 perturbation, atmospheric drag, SGP4 propagation) and their translation into engineering constraints on feature design; the practice of building diagnostics that can actually falsify a system's assumptions rather than confirming them, illustrated concretely by the live/train mismatch fix in Section 2.7.2; disciplined ML validation practice, including checkpoint selection and validation-set sizing; systems-level collaboration across a Python research codebase and a separate, higher-performance Rust simulator; and the experience of working through two consecutive architectural pivots within a single research cycle, and of representing that evolution honestly in a formal report rather than presenting only the final, cleaned-up state."),
  p("Areas identified for further development, carried forward as next steps, include completing the classical-baseline benchmark that the project's own success criteria depend on, gaining experience integrating two independently developed codebases (the Python model and the Rust simulator), and building the statistical evaluation practice needed to respond to small-sample results such as the 0/6 overturn-precision figure without either dismissing or over-interpreting them."),
]);

// =====================================================================
// CHAPTER 3 — CONCLUSION
// =====================================================================
const chapter3 = chapterSection("Chapter 3 \u2014 Conclusion", [
  h1("3. Conclusion"),
  p("Over approximately eight weeks, the team progressed from a literature review and a three-module proposal to a validated, end-to-end retrieval-and-GNN pipeline with real evaluation numbers, two genuinely fixed bugs backed by evidence, and an honest account of what those numbers do and do not support. The strongest output of the internship is not any single accuracy figure but the diagnostic discipline shown in Section 2.7.2: building a check specifically designed to catch the pipeline's own assumptions failing, finding that they had, and fixing the underlying cause rather than the symptom, twice."),
  p("The clearest gap between what was proposed and what was delivered is that the project has not yet closed the loop back to its own pre-agreed success criteria: the JPDA comparison that would turn the current strong retrieval numbers into an actual verdict on the core hypothesis has not been run, even though the classical baseline needed to run it \u2014 a working JPDA tracker \u2014 already exists in the Rust simulator. The two additional modules proposed at the outset, new-object track initiation and closed-loop sensor tasking, were not built within the internship period; this is a known and accepted trade-off, anticipated in the team's own mid-project contingency planning, rather than an unplanned shortfall."),
  p("Recommended next steps, in priority order, are: (1) run the head-to-head JPDA benchmark using the existing Rust simulator against the same synthetic data used to evaluate the Python pipeline; (2) reconnect the validated Python model to the Rust simulator so both classical and learned association can be exercised on one shared, higher-performance harness; (3) collect a larger sample of live GNN-overturn events to give the 0/6 result in Section 2.7.1 real statistical weight; and (4) if time allows, revisit the originally proposed track-initiation and sensor-tasking modules as extensions once the core comparison is complete."),
  p("For future PESURF cohorts undertaking a similarly exploratory research problem, our main recommendation is to schedule the classical-baseline comparison earlier in the timeline rather than treating it as a final step \u2014 in our case it remained available throughout but was consistently deprioritised in favour of building out the learned pipeline, which meant the project's central, falsifiable question was never actually answered within the internship period despite having all the pieces needed to answer it."),
  p("We are grateful to PES Research Foundation and to our mentor, Vandana M L, for the guidance, freedom, and critical feedback that made this iterative, occasionally uncomfortable process of finding and fixing our own mistakes possible, and to the PESURF programme for the opportunity to work on a real, externally posed research problem."),
]);

// =====================================================================
// CHAPTER 4 — REFERENCES
// =====================================================================
const chapter4 = chapterSection("Chapter 4 \u2014 References", [
  h1("4. References / Bibliography"),
  p("[1] B. Wang et al., \u201cResearch Advancements in Key Technologies for Space-Based Situational Awareness,\u201d Space Sci. Technol., 2022."),
  p("[2] A. K. Anil Kumar and B. Mukherjee, \u201cSpace Debris Mitigation Activities of ISRO,\u201d Trans. Indian Natl. Acad. Eng., 2025."),
  p("[3] B. Wei and B. D. Nener, \u201cMulti-Sensor Space Debris Tracking with Labeled RFS,\u201d IEEE Access, 2019."),
  p("[4] S. He, H.-S. Shin, and A. Tsourdos, \u201cTrajectory Optimization Using JPDA Filter,\u201d J. Guid. Control Dyn., 2020."),
  p("[5] B. Li et al., \u201cML-Based Approach for Improved Orbit Predictions,\u201d IEEE Trans. Aerosp. Electron. Syst., 2020."),
  p("[6] Y. Hu et al., \u201cCooperative Space Object Tracking via Universal Kalman Consensus Filter,\u201d Acta Astronaut., 2019."),
  p("[7] D. S. Roll et al., \u201cCosmosDSR \u2014 Automated Detection and Tracking of Orbital Debris,\u201d Adv. Space Res., 2023."),
  p("[8] N. Das et al., \u201cOptimal-Transport-Based Tracking of Space Objects,\u201d J. Guid. Control Dyn., 2019."),
  p("[9] M. Bolden et al., \u201cProbabilistic IOD in Cislunar Space Using Optical Sensors,\u201d AMOS Conference, 2022."),
  p("[10] T. Karthikeya et al., \u201cAI-Driven Space Debris Detection Using Deep Learning,\u201d IEEE ICSIMA, 2025."),
  p("[11] J. Li, X. Gao, and T. Jiang, \u201cGraph Networks for Multiple Object Tracking,\u201d WACV, 2020."),
  p("[12] X. Weng et al., \u201cGNN3DMOT: Graph Neural Network for 3D Multi-Object Tracking,\u201d CVPR, 2020."),
  p("[13] Y. Wang, K. Kitani, and X. Weng, \u201cJoint Object Detection and Multi-Object Tracking with GNNs,\u201d ICRA, 2021."),
]);

// =====================================================================
// APPENDIX
// =====================================================================
const appendix = chapterSection("Appendices", [
  h1("Appendices"),

  h2("Appendix A: Repository Reference"),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3000, 6360],
    rows: [
      new TableRow({ children: [cell("Python validation sprint", { bold: true, shade: "EDEDED", width: 3000 }), cell("github.com/SunOrangeBurger/leo-cat-tracker", { width: 6360 })] }),
      new TableRow({ children: [cell("Rust classical simulator", { bold: true, shade: "EDEDED", width: 3000 }), cell("github.com/SunOrangeBurger/rust-leo-sim", { width: 6360 })] }),
      new TableRow({ children: [cell("Literature review deck", { bold: true, shade: "EDEDED", width: 3000 }), cell("ISRO_SSA_LitReview_Combined_ELI5.pptx (June 2026)", { width: 6360 })] }),
      new TableRow({ children: [cell("Mid-project evaluation deck", { bold: true, shade: "EDEDED", width: 3000 }), cell("PES_RF_MidProject_Evaluation.pptx (July 2026)", { width: 6360 })] }),
    ],
  }),
  pRuns([run("")], { spacing: { before: 200 } }),

  h2("Appendix B: rust-leo-sim Repository Layout"),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3400, 5960],
    rows: [
      new TableRow({ children: [cell("Module", { bold: true, shade: "EDEDED", width: 3400 }), cell("Function", { bold: true, shade: "EDEDED", width: 5960 })] }),
      new TableRow({ children: [cell("objects.rs", { width: 3400 }), cell("Object pool, propagation, orbital parameters", { width: 5960 })] }),
      new TableRow({ children: [cell("sensor.rs", { width: 3400 }), cell("Ground stations, observations", { width: 5960 })] }),
      new TableRow({ children: [cell("tracker.rs", { width: 3400 }), cell("Nearest-neighbour tracker", { width: 5960 })] }),
      new TableRow({ children: [cell("jpda.rs", { width: 3400 }), cell("JPDA tracker (classical baseline)", { width: 5960 })] }),
      new TableRow({ children: [cell("ground_truth.rs", { width: 3400 }), cell("Ground truth table, metrics", { width: 5960 })] }),
      new TableRow({ children: [cell("hungarian.rs", { width: 3400 }), cell("Hungarian algorithm, OSPA metric", { width: 5960 })] }),
      new TableRow({ children: [cell("collision.rs", { width: 3400 }), cell("Collision detection", { width: 5960 })] }),
      new TableRow({ children: [cell("catalog.rs", { width: 3400 }), cell("Object catalog, CSV export", { width: 5960 })] }),
      new TableRow({ children: [cell("celestrak.rs", { width: 3400 }), cell("Live CelesTrak TLE ingestion", { width: 5960 })] }),
      new TableRow({ children: [cell("gui/", { width: 3400 }), cell("Unified configuration and simulation GUI", { width: 5960 })] }),
    ],
  }),
  pRuns([run("Table A.1: rust-leo-sim repository layout.", { italics: true, size: SZ_SMALL })], { alignment: AlignmentType.CENTER, spacing: { before: 100, after: 300 } }),

  h2("Appendix C: Reproducing the Reported Results"),
  p("The Section 2.7.1 results can be reproduced from the leo-cat-tracker repository with:"),
  new Paragraph({ spacing: { after: 120 }, children: [run("python run_full_evaluation.py", { font: "Courier New" })] }),
  p("This runs the full pipeline: catalog generation, embedding, GNN training, the tracking simulation, cross-checks, the live-vs-training comparison, and renders all evaluation plots. Run with --skip-generate --skip-embed to reuse an existing catalog and embeddings while only retraining and re-simulating, or --plots-only to re-render plots from a previous run's saved results. See run_full_evaluation.py --help for the full set of options."),

  h2("Appendix D: Progress Against the Original 12-Week Plan"),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [4200, 5160],
    rows: [
      new TableRow({ children: [cell("Workstream", { bold: true, shade: "1F3864", width: 4200 }), cell("Status", { bold: true, shade: "1F3864", width: 5160 })] }),
      new TableRow({ children: [cell("Literature review and proposal", { width: 4200 }), cell("Complete", { width: 5160 })] }),
      new TableRow({ children: [cell("Synthetic dataset generation", { width: 4200 }), cell("Complete", { width: 5160 })] }),
      new TableRow({ children: [cell("MLP object classifier (Phase 2)", { width: 4200 }), cell("Complete as scoped, superseded in final pipeline", { width: 5160 })] }),
      new TableRow({ children: [cell("Class-specific GNNs (Phase 2)", { width: 4200 }), cell("Superseded by one class-agnostic GNN", { width: 5160 })] }),
      new TableRow({ children: [cell("Unified GNN association engine (final)", { width: 4200 }), cell("Complete and validated end-to-end", { width: 5160 })] }),
      new TableRow({ children: [cell("ANN retrieval gate", { width: 4200 }), cell("Complete, 100% candidate recall", { width: 5160 })] }),
      new TableRow({ children: [cell("Live/train mismatch diagnostic", { width: 4200 }), cell("Complete, found and fixed a real bug", { width: 5160 })] }),
      new TableRow({ children: [cell("Rust orbital simulator (classical baseline)", { width: 4200 }), cell("Complete; not yet connected to the Python model", { width: 5160 })] }),
      new TableRow({ children: [cell("JPDA head-to-head benchmark", { width: 4200 }), cell("Not started", { width: 5160 })] }),
      new TableRow({ children: [cell("Module B \u2014 new-object track initiation", { width: 4200 }), cell("Not started as originally scoped", { width: 5160 })] }),
      new TableRow({ children: [cell("Module C \u2014 DRL sensor-tasking feedback loop", { width: 4200 }), cell("Not started", { width: 5160 })] }),
      new TableRow({ children: [cell("Python model \u2013 Rust simulator integration", { width: 4200 }), cell("Not started", { width: 5160 })] }),
    ],
  }),
]);

// =====================================================================
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: SZ }, paragraph: { spacing: LINE15, alignment: AlignmentType.JUSTIFIED } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: SZ, bold: true, underline: {} }, paragraph: { alignment: AlignmentType.LEFT } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: SZ, bold: true, underline: {} }, paragraph: { alignment: AlignmentType.LEFT } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: SZ, bold: true }, paragraph: { alignment: AlignmentType.LEFT } },
    ],
  },
  sections: [frontMatter, chapter1, chapter2, chapter3, chapter4, appendix],
});

Packer.toBuffer(doc).then((buf) => {
  require("fs").writeFileSync("PESURF_Internship_Report.docx", buf);0
  console.log("written");
});