/* Catches the one typographic bug this codebase actually produces.
 *
 * Astro follows JSX whitespace rules: a line break between a word and
 * whatever follows it is not a space. So
 *
 *     It suggests
 *     {top[0]}, then …
 *
 * renders as "It suggestssalt", and
 *
 *     At this sample size it is
 *     <b>indistinguishable</b>
 *
 * renders as "it isindistinguishable". The failure is invisible in the
 * source, survives review because the source looks correct, and shows up
 * only as a missing space on the rendered page. Seven of these shipped
 * before anyone read a page closely enough to catch one.
 *
 * Four directions, since the boundary can face either way:
 *   word -> {value}      word -> <inline>
 *   {value} -> word      </inline> -> word
 *
 * The rule: when a sentence continues across a line break into something
 * that is not plain text, the author writes the space explicitly as {" "}.
 *
 * Conditional expressions are exempt. A ternary or a `&&` render controls
 * its own leading space inside each branch — `{n === 1 ? " is" : " are"}`
 * is correct as written — and flagging those would train people to silence
 * the check rather than read it.
 */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";

const DIRS = ["src/pages", "src/components", "src/layouts"];

/* Elements that sit inside a sentence. A block element on its own line is
 * a new paragraph, not a missing space, so only these are checked. */
const INLINE = "a|abbr|b|cite|code|em|i|kbd|mark|q|s|small|span|strong|sub|sup|u|var|time";

/* Components of this site that render inline. A capitalised name could be
 * either — <Demo /> is a block and <Num /> is a span — and the difference
 * decides whether a full stop before it is the end of a paragraph or the
 * middle of one. Anything not listed here is assumed to be a block. */
const INLINE_COMPONENTS = "Num|Fragment";

/** Opens a plain value interpolation: `{name`, `{fn(`, `{obj.key`. */
const OPENS_VALUE = /^\s*\{[A-Za-z_$][\w$]*(?:[.[(]|\s*\})/;
/** Opens an inline element, or any component (which may render inline). */
const OPENS_INLINE = new RegExp(`^\\s*<(?:${INLINE}|[A-Z][A-Za-z0-9]*)[\\s/>]`);
/** Opens something certainly inline, so a full stop before it is mid-paragraph
 *  rather than the end of one. */
const OPENS_CERTAINLY_INLINE =
  new RegExp(`^\\s*<(?:${INLINE}|${INLINE_COMPONENTS})[\\s/>]`);
/** Closes an interpolation or an inline element at end of line. */
const CLOSES_VALUE = /\}\s*$/;
const CLOSES_INLINE = new RegExp(`(</(?:${INLINE}|[A-Z][A-Za-z0-9]*)>|/>)\\s*$`);
/** An explicit space is the fix, not the fault. */
const EXPLICIT_SPACE = /\{\s*["'] ["']\s*\}\s*$/;
/** Continues a sentence: a word character or mid-sentence punctuation. */
const CONTINUES = /[\w,;:)]\s*$/;
const SENTENCE_END = /[.!?]["')\]]?\s*$/;
/** Next line resumes prose rather than opening new markup. */
const RESUMES_TEXT = /^\s*[a-z]/;
/** A branchy expression whose spacing lives inside its own branches. */
const CONDITIONAL = /[?&|]{2}|\?\s|\s:\s/;
const NOT_PROSE =
  /^\s*(\/\/|\/\*|\*|const |let |var |import |export |type |interface |function )/;
/** True when the line stops part-way through a tag, so the break sits
 *  between attributes rather than between words. */
const midTag = (line) => line.lastIndexOf("<") > line.lastIndexOf(">");

const FIX = process.argv.includes("--fix");
const found = [];

/* Frontmatter, <style> and <script> are code, not prose. A CSS rule ends
 * in `}` and the next line starts with a selector, which is the same shape
 * as a swallowed space and none of it is rendered as text. */
const proseMask = (lines) => {
  const prose = new Array(lines.length).fill(true);
  let fence = 0;
  let block = null;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (fence < 2 && /^---\s*$/.test(line)) {
      fence++;
      prose[i] = false;
      continue;
    }
    if (fence === 1) prose[i] = false;
    if (block) {
      prose[i] = false;
      if (new RegExp(`</${block}>`, "i").test(line)) block = null;
      continue;
    }
    const open = /<(style|script)[\s>]/i.exec(line);
    if (open) {
      prose[i] = false;
      /* A self-closing <script ... /> is Astro's JSON-island pattern and
       * closes on its own line. Treating it as an opening tag masked the
       * whole rest of the file as code and silently switched this check
       * off, which is how a missing space shipped in Predict.astro. */
      const selfClosed = /\/>\s*$/.test(line);
      const closed = new RegExp(`</${open[1]}>`, "i").test(line);
      if (!selfClosed && !closed) block = open[1].toLowerCase();
    }
  }
  return prose;
};

/* True for lines sitting between a `<Tag` and its closing `>`, so a break
 * between two attributes is not read as a break between two words. midTag
 * only sees one line; an attribute list spans several. */
const attrMask = (lines) => {
  const inAttrs = new Array(lines.length).fill(false);
  let open = false;
  for (let i = 0; i < lines.length; i++) {
    inAttrs[i] = open;
    const lt = lines[i].lastIndexOf("<");
    const gt = lines[i].lastIndexOf(">");
    if (lt > gt) open = true;
    else if (gt > lt) open = false;
  }
  return inAttrs;
};

const check = (path, lines) => {
  const prose = proseMask(lines);
  const inAttrs = attrMask(lines);
  for (let i = 0; i < lines.length - 1; i++) {
    if (!prose[i] || !prose[i + 1]) continue;
    const here = lines[i];
    const next = lines[i + 1];
    if (NOT_PROSE.test(here) || NOT_PROSE.test(next)) continue;
    if (midTag(here) || inAttrs[i] || EXPLICIT_SPACE.test(here)) continue;

    let why = null;
    const ends = SENTENCE_END.test(here);
    if (CONTINUES.test(here) && !ends) {
      if (OPENS_VALUE.test(next) && !CONDITIONAL.test(next)) why = "an interpolation";
      else if (OPENS_INLINE.test(next)) why = "an inline element";
    } else if (ends && OPENS_CERTAINLY_INLINE.test(next)) {
      // A new sentence in the same paragraph still needs the space before it.
      why = "an inline element";
    } else if (RESUMES_TEXT.test(next)) {
      if (CLOSES_VALUE.test(here) && !CONDITIONAL.test(here)) why = "text after an interpolation";
      else if (CLOSES_INLINE.test(here)) why = "text after an inline element";
    }
    if (why) found.push({ path, line: i + 1, why, here: here.trim(), next: next.trim() });
  }
};

for (const dir of DIRS) {
  for (const file of readdirSync(dir)) {
    if (!/\.(astro|ts|tsx)$/.test(file)) continue;
    check(`${dir}/${file}`, readFileSync(`${dir}/${file}`, "utf8").split("\n"));
  }
}

/* Every direction repairs the same way: the space belongs at the end of
 * the line that opens the break, so the source shows what will render. */
if (FIX && found.length) {
  const byFile = new Map();
  for (const f of found) {
    if (!byFile.has(f.path)) byFile.set(f.path, []);
    byFile.get(f.path).push(f.line - 1);
  }
  for (const [path, idx] of byFile) {
    const lines = readFileSync(path, "utf8").split("\n");
    for (const i of idx) lines[i] = lines[i].replace(/\s*$/, '{" "}');
    writeFileSync(path, lines.join("\n"));
    console.log(`  fixed ${idx.length}  ${path}`);
  }
  console.log(`\n${found.length} space(s) made explicit. Re-run to confirm.`);
  process.exit(0);
}

for (const f of found) {
  console.log(`  FAIL  ${f.path}:${f.line}  no space before ${f.why}`);
  console.log(`          ...${f.here.slice(-58)}`);
  console.log(`          ${f.next.slice(0, 58)}...`);
}

if (found.length) {
  console.log(
    `\n${found.length} missing space(s). End the line with {" "} ` +
      `so the rendered text keeps the space its source implies.`,
  );
  process.exit(1);
}

console.log("  ok    no line break silently swallows a space");
console.log("\nOK — copy renders with the spaces its source implies.");
