import fs from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";

const projectRoot = "/Users/simon/Documents/data_science /big data prog";
const threadId = process.env.CODEX_THREAD_ID || `manual-${new Date().toISOString().replace(/[-:T.Z]/g, "").slice(0, 14)}-bdppt`;
const workspace = path.join(projectRoot, "outputs", threadId, "presentations", "italian-labour-forecast");
const slidesDir = path.join(workspace, "slides");
const previewDir = path.join(workspace, "preview");
const layoutDir = path.join(workspace, "layout", "final");
const qaDir = path.join(workspace, "qa");
const outputDir = path.join(projectRoot, "deliverables");
const finalPptx = path.join(outputDir, "italian_labour_forecast_policy_deck.pptx");
const skillDir = "/Users/simon/.codex/plugins/cache/openai-primary-runtime/presentations/26.521.10419/skills/presentations";
const nodeBin = "/Users/simon/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node";

const figures = {
  regionalRankings: path.join(projectRoot, "reports/figures/regional_rankings_latest_year.png"),
  labourTrends: path.join(projectRoot, "reports/figures/labour_trends.png"),
  correlation: path.join(projectRoot, "reports/figures/correlation_heatmap.png"),
  gdpTargets: path.join(projectRoot, "reports/figures/gdp_vs_targets_latest_year.png"),
  forecastMae: path.join(projectRoot, "reports/figures/forecast_mae_comparison.png"),
  rollingBacktest: path.join(projectRoot, "reports/figures/rolling_backtest_mae.png"),
  featureImportance: path.join(projectRoot, "reports/figures/rf_feature_importance_unemployment.png"),
  policyQuadrants: path.join(projectRoot, "reports/figures/policy_gdp_unemployment_quadrants.png"),
  policyRanking: path.join(projectRoot, "reports/figures/policy_priority_ranking_2025.png"),
  criticalProb: path.join(projectRoot, "reports/figures/critical_region_probability_2025.png"),
};

function jsString(value) {
  return JSON.stringify(value);
}

async function write(filePath, content) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, content, "utf8");
}

const common = `
export const C = {
  navy: "#16213E",
  ink: "#1F2937",
  muted: "#6B7280",
  bg: "#F7F8F4",
  panel: "#FFFFFF",
  teal: "#2A9D8F",
  green: "#6A994E",
  coral: "#E76F51",
  gold: "#E9C46A",
  blue: "#457B9D",
  violet: "#8E6C8A",
  line: "#D8DDD2",
};

export function base(slide, ctx, section = "") {
  ctx.addShape(slide, { x: 0, y: 0, w: ctx.W, h: ctx.H, fill: C.bg, line: ctx.line() });
  ctx.addShape(slide, { x: 0, y: 0, w: ctx.W, h: 12, fill: C.teal, line: ctx.line() });
  ctx.addText(slide, {
    text: section.toUpperCase(),
    x: 52, y: 28, w: 390, h: 22,
    size: 13, color: C.teal, bold: true,
    typeface: ctx.fonts.body,
  });
  ctx.addText(slide, {
    text: "Big Data project | ISTAT regional labour market + GDP",
    x: 850, y: 678, w: 360, h: 22,
    size: 11, color: C.muted, align: "right",
  });
}

export function title(slide, ctx, text, subtitle = "") {
  ctx.addText(slide, {
    text, x: 52, y: 58, w: 830, h: 92,
    size: 34, color: C.navy, bold: true, typeface: ctx.fonts.title,
  });
  if (subtitle) {
    ctx.addText(slide, { text: subtitle, x: 54, y: 143, w: 820, h: 42, size: 16, color: C.muted });
  }
}

export function claim(slide, ctx, text) {
  ctx.addText(slide, {
    text, x: 55, y: 580, w: 1120, h: 58,
    size: 20, color: C.navy, bold: true, typeface: ctx.fonts.title,
  });
}

export function kpi(slide, ctx, { x, y, w = 210, h = 116, value, label, note, color = C.teal }) {
  ctx.addShape(slide, { x, y, w, h, fill: C.panel, line: { style: "solid", fill: C.line, width: 1 } });
  ctx.addShape(slide, { x, y, w: 8, h, fill: color, line: ctx.line() });
  ctx.addText(slide, { text: value, x: x + 20, y: y + 16, w: w - 35, h: 40, size: 32, color, bold: true, typeface: ctx.fonts.title });
  ctx.addText(slide, { text: label, x: x + 22, y: y + 61, w: w - 35, h: 24, size: 14, color: C.ink, bold: true });
  if (note) ctx.addText(slide, { text: note, x: x + 22, y: y + 87, w: w - 35, h: 22, size: 11, color: C.muted });
}

export function bullet(slide, ctx, items, x, y, w, options = {}) {
  const size = options.size || 18;
  const color = options.color || C.ink;
  items.forEach((item, i) => {
    const yy = y + i * (options.gap || 46);
    ctx.addShape(slide, { x, y: yy + 7, w: 9, h: 9, fill: options.dot || C.teal, line: ctx.line() });
    ctx.addText(slide, { text: item, x: x + 25, y: yy, w, h: 38, size, color });
  });
}

export async function chart(slide, ctx, imagePath, x, y, w, h, caption = "") {
  ctx.addShape(slide, { x, y, w, h, fill: C.panel, line: { style: "solid", fill: C.line, width: 1 } });
  await ctx.addImage(slide, { path: imagePath, x: x + 8, y: y + 8, w: w - 16, h: h - (caption ? 42 : 16), fit: "contain" });
  if (caption) ctx.addText(slide, { text: caption, x: x + 18, y: y + h - 30, w: w - 36, h: 20, size: 11, color: C.muted });
}

export function miniTable(slide, ctx, rows, x, y, colWidths, rowH = 36) {
  rows.forEach((row, r) => {
    let xx = x;
    row.forEach((cell, c) => {
      const fill = r === 0 ? C.navy : (r % 2 === 0 ? "#F0F4EF" : C.panel);
      const color = r === 0 ? "#FFFFFF" : C.ink;
      ctx.addShape(slide, { x: xx, y: y + r * rowH, w: colWidths[c], h: rowH, fill, line: { style: "solid", fill: C.line, width: 1 } });
      ctx.addText(slide, { text: String(cell), x: xx + 9, y: y + r * rowH + 8, w: colWidths[c] - 18, h: rowH - 10, size: r === 0 ? 12 : 13, color, bold: r === 0 });
      xx += colWidths[c];
    });
  });
}

export function sectionNumber(slide, ctx, n, label, x, y, color = C.teal) {
  ctx.addText(slide, { text: String(n).padStart(2, "0"), x, y, w: 92, h: 70, size: 52, color, bold: true, typeface: ctx.fonts.title });
  ctx.addText(slide, { text: label, x: x + 92, y: y + 12, w: 410, h: 50, size: 22, color: C.navy, bold: true, typeface: ctx.fonts.title });
}
`;

function slideModule(n, body) {
  const id = String(n).padStart(2, "0");
  return `import { base, title, claim, kpi, bullet, chart, miniTable, sectionNumber, C } from "./common.mjs";

export async function slide${id}(presentation, ctx) {
  const slide = presentation.slides.add();
${body}
  return slide;
}
`;
}

const slides = [
  slideModule(1, `
  base(slide, ctx, "Executive summary");
  ctx.addShape(slide, { x: 0, y: 12, w: 1280, h: 708, fill: C.navy, line: ctx.line() });
  ctx.addShape(slide, { x: 0, y: 12, w: 1280, h: 708, fill: "linear(10deg, #16213E 0%, #2A9D8F 100%)", line: ctx.line() });
  ctx.addText(slide, { text: "Prevedere disoccupazione e occupazione in Italia", x: 70, y: 82, w: 840, h: 120, size: 44, color: "#FFFFFF", bold: true, typeface: ctx.fonts.title });
  ctx.addText(slide, { text: "Dati ISTAT regionali, PIL come fattore esterno e modelli OLS/ML per trasformare il forecast in priorità di intervento", x: 74, y: 220, w: 760, h: 60, size: 20, color: "#E6F4F1" });
  kpi(slide, ctx, { x: 74, y: 360, w: 250, value: "0.80 p.p.", label: "errore medio disoccupazione", note: "punti percentuali | Random Forest", color: "#E9C46A" });
  kpi(slide, ctx, { x: 344, y: 360, w: 250, value: "15.4k", label: "errore medio occupati", note: "migliaia di persone | lag-1", color: "#457B9D" });
  kpi(slide, ctx, { x: 614, y: 360, w: 250, value: "F1 0.92", label: "regioni critiche", note: "classificazione | Logistic Regression", color: "#E76F51" });
  ctx.addText(slide, { text: "Messaggio chiave", x: 900, y: 112, w: 260, h: 28, size: 16, color: "#E9C46A", bold: true });
  ctx.addText(slide, { text: "Il ML individua meglio le aree a rischio, ma la risposta deve essere territoriale: investimenti produttivi, competenze tecniche e politiche mirate sui NEET.", x: 900, y: 150, w: 300, h: 180, size: 23, color: "#FFFFFF", bold: true, typeface: ctx.fonts.title });
  ctx.addText(slide, { text: "Regioni prioritarie: Campania, Sicilia, Calabria, Puglia, Sardegna.", x: 902, y: 500, w: 285, h: 54, size: 16, color: "#E6F4F1" });
  `),
  slideModule(2, `
  base(slide, ctx, "Narrative spine");
  title(slide, ctx, "Una lettura lineare: dai dati alla decisione", "La presentazione segue il percorso analitico del progetto e chiude con raccomandazioni operative.");
  sectionNumber(slide, ctx, 1, "EDA: differenze territoriali e indicatori chiave", 80, 215, C.teal);
  sectionNumber(slide, ctx, 2, "Forecasting: previsione con feature laggate", 80, 315, C.blue);
  sectionNumber(slide, ctx, 3, "Confronto OLS vs ML: quando la complessità serve", 80, 415, C.coral);
  sectionNumber(slide, ctx, 4, "Policy insights: regioni prioritarie e leve di intervento", 650, 215, C.green);
  sectionNumber(slide, ctx, 5, "Classificazione criticità: strumento decisionale", 650, 315, C.violet);
  sectionNumber(slide, ctx, 6, "Conclusione: cosa fare per migliorare l’occupazione", 650, 415, C.gold);
  claim(slide, ctx, "Obiettivo: non solo prevedere valori, ma capire dove intervenire e con quali strumenti.");
  `),
  slideModule(3, `
  base(slide, ctx, "Dati e metodologia");
  title(slide, ctx, "Il panel collega mercato del lavoro, PIL e fragilità giovanile", "Unità di analisi: regione/provincia autonoma per anno; modello supervisionato con variabili dell’anno precedente.");
  kpi(slide, ctx, { x: 70, y: 210, value: "21", label: "territori", note: "regioni + province autonome", color: C.teal });
  kpi(slide, ctx, { x: 310, y: 210, value: "2018–2025", label: "target lavoro", note: "disoccupazione, occupati, NEET", color: C.blue });
  kpi(slide, ctx, { x: 550, y: 210, value: "2018–2024", label: "campione con PIL", note: "PIL disponibile fino al 2024", color: C.coral });
  kpi(slide, ctx, { x: 790, y: 210, value: "147", label: "osservazioni forecast", note: "21 territori x 7 anni", color: C.green });
  bullet(slide, ctx, [
    "Target: tasso di disoccupazione e occupati in migliaia",
    "Predittori: lag dei target, PIL, PIL per occupato, NEET, territorio",
    "Split temporale: si allena sul passato e si testa sugli anni successivi",
    "Policy layer: ranking delle aree dove il rischio occupazionale è più alto"
  ], 110, 395, 920, { size: 19, gap: 44, dot: C.teal });
  claim(slide, ctx, "La scelta metodologica evita leakage: il 2025 viene previsto usando informazioni disponibili al 2024.");
  `),
  slideModule(4, `
  base(slide, ctx, "EDA");
  title(slide, ctx, "La disoccupazione 2024 è concentrata nel Mezzogiorno", "Le differenze territoriali sono il primo segnale: il problema non è omogeneo sul territorio nazionale.");
  await chart(slide, ctx, ${jsString(figures.regionalRankings)}, 58, 168, 760, 380, "Classifica regionale 2024: disoccupazione e occupati.");
  kpi(slide, ctx, { x: 860, y: 190, w: 270, value: "15.90%", label: "Campania", note: "disoccupazione 2024", color: C.coral });
  kpi(slide, ctx, { x: 860, y: 330, w: 270, value: "13.42%", label: "Calabria", note: "disoccupazione 2024", color: C.coral });
  kpi(slide, ctx, { x: 860, y: 470, w: 270, value: "4.41M", label: "Lombardia", note: "occupati 2024", color: C.teal });
  claim(slide, ctx, "Il ranking indica una frattura territoriale: alta disoccupazione al Sud, massa occupazionale più forte al Nord.");
  `),
  slideModule(5, `
  base(slide, ctx, "EDA");
  title(slide, ctx, "PIL, occupazione e NEET misurano dimensioni diverse del problema", "Il PIL assoluto spiega la scala economica; NEET e PIL per occupato aiutano a leggere la fragilità.");
  await chart(slide, ctx, ${jsString(figures.correlation)}, 65, 168, 520, 360, "Correlazioni tra variabili principali.");
  await chart(slide, ctx, ${jsString(figures.gdpTargets)}, 625, 168, 570, 360, "PIL per occupato vs disoccupazione; PIL assoluto vs occupati.");
  kpi(slide, ctx, { x: 85, y: 550, w: 210, value: "0.98", label: "PIL ↔ occupati", note: "correlazione", color: C.teal });
  kpi(slide, ctx, { x: 330, y: 550, w: 210, value: "0.96", label: "NEET ↔ disoccupazione", note: "correlazione", color: C.coral });
  kpi(slide, ctx, { x: 575, y: 550, w: 210, value: "-0.67", label: "PIL/occupato ↔ disoccupazione", note: "relazione negativa", color: C.blue });
  `),
  slideModule(6, `
  base(slide, ctx, "Forecasting design");
  title(slide, ctx, "Il forecast usa solo informazioni disponibili nell’anno precedente", "La logica è progettata per rispondere alla domanda: cosa posso sapere oggi per stimare il rischio domani?");
  ctx.addShape(slide, { x: 90, y: 232, w: 250, h: 135, fill: C.panel, line: { style: "solid", fill: C.line, width: 1 } });
  ctx.addText(slide, { text: "Input t-1", x: 120, y: 255, w: 200, h: 35, size: 27, color: C.teal, bold: true, typeface: ctx.fonts.title });
  bullet(slide, ctx, ["disoccupazione", "occupati", "PIL e PIL/occupato", "NEET"], 118, 305, 190, { size: 14, gap: 22 });
  ctx.addShape(slide, { x: 395, y: 278, w: 160, h: 24, fill: C.teal, line: ctx.line() });
  ctx.addShape(slide, { x: 540, y: 266, w: 34, h: 48, fill: C.teal, line: ctx.line(), geometry: "triangle" });
  ctx.addShape(slide, { x: 610, y: 212, w: 260, h: 175, fill: C.panel, line: { style: "solid", fill: C.line, width: 1 } });
  ctx.addText(slide, { text: "Modelli", x: 642, y: 236, w: 200, h: 35, size: 27, color: C.blue, bold: true, typeface: ctx.fonts.title });
  bullet(slide, ctx, ["Naive lag-1", "OLS classica", "Ridge / ElasticNet", "Random Forest / GB"], 638, 288, 205, { size: 14, gap: 23, dot: C.blue });
  ctx.addShape(slide, { x: 920, y: 232, w: 250, h: 135, fill: C.panel, line: { style: "solid", fill: C.line, width: 1 } });
  ctx.addText(slide, { text: "Output t", x: 952, y: 255, w: 200, h: 35, size: 27, color: C.coral, bold: true, typeface: ctx.fonts.title });
  bullet(slide, ctx, ["disoccupazione prevista", "occupati previsti", "probabilità criticità", "priorità policy"], 948, 305, 190, { size: 14, gap: 22, dot: C.coral });
  claim(slide, ctx, "La previsione diventa utile quando si traduce in ranking delle regioni e in azioni occupazionali.");
  `),
  slideModule(7, `
  base(slide, ctx, "Forecasting");
  title(slide, ctx, "Per la disoccupazione il ML non lineare supera OLS e baseline", "Nel test 2024–2025 Random Forest riduce l’errore rispetto alla regressione classica.");
  miniTable(slide, ctx, [
    ["Modello", "MAE", "RMSE", "R²"],
    ["Random Forest", "0.72 pp", "0.95", "0.92"],
    ["Gradient Boosting", "0.74 pp", "0.95", "0.92"],
    ["ElasticNet ML lineare", "0.85 pp", "1.01", "0.92"],
    ["Ridge ML lineare", "0.88 pp", "1.05", "0.91"],
    ["OLS classica", "0.91 pp", "1.20", "0.88"],
    ["Naive lag-1", "0.96 pp", "1.24", "0.87"],
  ], 80, 190, [290, 120, 120, 100], 44);
  kpi(slide, ctx, { x: 770, y: 205, w: 290, value: "−21%", label: "errore vs OLS", note: "MAE Random Forest rispetto a OLS", color: C.teal });
  kpi(slide, ctx, { x: 770, y: 350, w: 290, value: "0.72 pp", label: "errore medio", note: "test 2024–2025", color: C.coral });
  ctx.addText(slide, { text: "Interpretazione", x: 770, y: 480, w: 300, h: 26, size: 16, color: C.navy, bold: true });
  ctx.addText(slide, { text: "La disoccupazione non è descritta bene da una relazione lineare semplice: territorio, persistenza e NEET interagiscono.", x: 770, y: 506, w: 330, h: 54, size: 15, color: C.ink });
  claim(slide, ctx, "Il ML serve davvero per prevedere la disoccupazione, ma il vantaggio va letto come supporto alla decisione territoriale.");
  `),
  slideModule(8, `
  base(slide, ctx, "Forecasting");
  title(slide, ctx, "Per gli occupati vince la persistenza: l’anno precedente basta spesso", "Il numero di occupati regionali cambia lentamente; la baseline lag-1 batte modelli più complessi.");
  miniTable(slide, ctx, [
    ["Modello", "MAE", "MAPE", "R²"],
    ["Naive lag-1", "12.76k", "1.16%", "0.9996"],
    ["Gradient Boosting", "29.70k", "3.74%", "0.9978"],
    ["Ridge ML lineare", "31.94k", "7.45%", "0.9986"],
    ["OLS classica", "32.45k", "9.43%", "0.9985"],
    ["Random Forest", "57.32k", "6.08%", "0.9896"],
  ], 78, 192, [300, 120, 120, 110], 48);
  await chart(slide, ctx, ${jsString(figures.forecastMae)}, 690, 170, 500, 345, "Confronto MAE test 2024–2025.");
  claim(slide, ctx, "Risultato importante: il Machine Learning non è sempre superiore; per variabili molto persistenti la baseline è il benchmark da battere.");
  `),
  slideModule(9, `
  base(slide, ctx, "Backtesting");
  title(slide, ctx, "Il backtesting conferma la robustezza della lettura", "Su tre finestre temporali, Random Forest resta migliore per disoccupazione; lag-1 resta migliore per occupati.");
  await chart(slide, ctx, ${jsString(figures.rollingBacktest)}, 65, 170, 790, 390, "MAE medio su finestre rolling.");
  kpi(slide, ctx, { x: 900, y: 190, w: 245, value: "0.80 pp", label: "RF disoccupazione", note: "MAE medio rolling", color: C.teal });
  kpi(slide, ctx, { x: 900, y: 330, w: 245, value: "1.37 pp", label: "OLS disoccupazione", note: "MAE medio rolling", color: C.coral });
  kpi(slide, ctx, { x: 900, y: 470, w: 245, value: "15.4k", label: "lag-1 occupati", note: "MAE medio rolling", color: C.blue });
  claim(slide, ctx, "La validazione temporale sostiene il messaggio: complessità utile sulla disoccupazione, persistenza dominante sugli occupati.");
  `),
  slideModule(10, `
  base(slide, ctx, "Explainability");
  title(slide, ctx, "La disoccupazione dell’anno precedente domina la previsione", "La persistenza storica è il driver principale; il NEET è il secondo segnale informativo.");
  await chart(slide, ctx, ${jsString(figures.featureImportance)}, 70, 175, 710, 370, "Feature importance Random Forest.");
  kpi(slide, ctx, { x: 835, y: 205, w: 260, value: "0.93", label: "disoccupazione lag-1", note: "importanza relativa", color: C.teal });
  kpi(slide, ctx, { x: 835, y: 345, w: 260, value: "0.054", label: "NEET lag-1", note: "secondo predittore", color: C.coral });
  ctx.addText(slide, { text: "Nota interpretativa", x: 835, y: 492, w: 260, h: 25, size: 16, color: C.navy, bold: true });
  ctx.addText(slide, { text: "Il PIL pesa meno nella previsione puntuale, ma resta decisivo per leggere la qualità economica del territorio.", x: 835, y: 518, w: 310, h: 58, size: 15, color: C.ink });
  ctx.addText(slide, { text: "Il modello dice cosa predice; la policy spiega perché il rischio si concentra e come ridurlo.", x: 55, y: 616, w: 1060, h: 38, size: 20, color: C.navy, bold: true, typeface: ctx.fonts.title });
  `),
  slideModule(11, `
  base(slide, ctx, "Policy insights");
  title(slide, ctx, "Le regioni prioritarie combinano disoccupazione alta e PIL per occupato basso", "Il grafico a quadranti traduce il forecast in una mappa decisionale.");
  await chart(slide, ctx, ${jsString(figures.policyQuadrants)}, 58, 165, 800, 420, "Dimensione dei punti = disoccupati stimati.");
  miniTable(slide, ctx, [
    ["Regione", "Disocc. prev.", "NEET", "PIL/occup."],
    ["Campania", "14.96%", "24.90%", "82.6k"],
    ["Sicilia", "13.03%", "25.74%", "78.3k"],
    ["Calabria", "13.07%", "26.23%", "76.5k"],
    ["Puglia", "8.85%", "21.42%", "74.6k"],
    ["Sardegna", "7.46%", "17.79%", "76.0k"],
  ], 885, 190, [126, 96, 72, 92], 42);
  claim(slide, ctx, "Quando disoccupazione prevista, NEET e basso PIL per occupato si sovrappongono, serve una risposta integrata lavoro–sviluppo.");
  `),
  slideModule(12, `
  base(slide, ctx, "Policy priority");
  title(slide, ctx, "La priorità 2025 si concentra soprattutto nel Mezzogiorno", "Il ranking combina rischio di disoccupazione, NEET, PIL per occupato e massa stimata di disoccupati.");
  await chart(slide, ctx, ${jsString(figures.policyRanking)}, 70, 165, 640, 405, "Top regioni per risk score.");
  kpi(slide, ctx, { x: 765, y: 185, w: 260, value: "292k", label: "Campania", note: "disoccupati stimati dal forecast", color: C.coral });
  kpi(slide, ctx, { x: 765, y: 325, w: 260, value: "214k", label: "Sicilia", note: "disoccupati stimati dal forecast", color: C.coral });
  kpi(slide, ctx, { x: 765, y: 465, w: 260, value: "123k", label: "Puglia", note: "disoccupati stimati dal forecast", color: C.coral });
  claim(slide, ctx, "La policy dovrebbe partire dalle aree dove il rischio è alto e la capacità produttiva locale è più fragile.");
  `),
  slideModule(13, `
  base(slide, ctx, "Criticality classification");
  title(slide, ctx, "La classificazione rende il modello uno strumento decisionale", "Invece di prevedere solo un valore, il modello segnala quali regioni rischiano di essere critiche.");
  await chart(slide, ctx, ${jsString(figures.criticalProb)}, 70, 170, 660, 365, "Probabilità di regione critica nel 2025.");
  kpi(slide, ctx, { x: 780, y: 185, w: 245, value: "0.93", label: "Accuracy", note: "Logistic Regression", color: C.teal });
  kpi(slide, ctx, { x: 780, y: 325, w: 245, value: "0.92", label: "F1-score", note: "regioni critiche", color: C.blue });
  kpi(slide, ctx, { x: 780, y: 465, w: 245, value: "0.97", label: "ROC AUC", note: "separazione classi", color: C.coral });
  claim(slide, ctx, "La Logistic Regression è abbastanza interpretabile e funziona bene: utile come early-warning semplice per le regioni critiche.");
  `),
  slideModule(14, `
  base(slide, ctx, "Recommendations");
  title(slide, ctx, "Cinque leve per migliorare l’occupazione nelle aree prioritarie", "Le raccomandazioni seguono direttamente dai cluster: alto rischio, NEET elevato, PIL per occupato basso.");
  const recs = [
    ["1", "Investimenti produttivi mirati", "Attrarre imprese e filiere nei territori con basso PIL per occupato."],
    ["2", "Formazione tecnica e ITS", "Allineare competenze ai settori con domanda reale di lavoro."],
    ["3", "Apprendistato e NEET", "Ridurre la distanza giovani–mercato con percorsi scuola-lavoro regionali."],
    ["4", "Centri per l’impiego potenziati", "Matching più rapido su profili, imprese e fabbisogni locali."],
    ["5", "Monitoraggio annuale", "Valutare NEET, disoccupazione prevista e PIL per occupato, non solo occupati totali."],
  ];
  recs.forEach((r, i) => {
    const y = 170 + i * 82;
    ctx.addShape(slide, { x: 82, y, w: 52, h: 52, fill: i < 3 ? C.teal : C.blue, line: ctx.line() });
    ctx.addText(slide, { text: r[0], x: 82, y: y + 8, w: 52, h: 38, size: 26, color: "#FFFFFF", bold: true, align: "center", typeface: ctx.fonts.title });
    ctx.addText(slide, { text: r[1], x: 158, y: y - 2, w: 470, h: 28, size: 21, color: C.navy, bold: true, typeface: ctx.fonts.title });
    ctx.addText(slide, { text: r[2], x: 158, y: y + 30, w: 760, h: 32, size: 16, color: C.ink });
  });
  ctx.addShape(slide, { x: 955, y: 188, w: 210, h: 290, fill: "#EAF5F2", line: { style: "solid", fill: "#BFD8D2", width: 1 } });
  ctx.addText(slide, { text: "Priorità", x: 980, y: 216, w: 160, h: 28, size: 22, color: C.teal, bold: true, typeface: ctx.fonts.title });
  ctx.addText(slide, { text: "Campania\\nSicilia\\nCalabria\\nPuglia\\nSardegna", x: 982, y: 265, w: 160, h: 150, size: 22, color: C.navy, bold: true, typeface: ctx.fonts.title });
  claim(slide, ctx, "La risposta più efficace combina domanda di lavoro, competenze e riduzione dei NEET.");
  `),
  slideModule(15, `
  base(slide, ctx, "Final conclusion");
  title(slide, ctx, "Conclusione: prevedere è utile solo se orienta l’intervento", "Il progetto passa da dati ufficiali a modelli predittivi e chiude con priorità territoriali operative.");
  kpi(slide, ctx, { x: 75, y: 185, w: 260, value: "RF", label: "migliore per disoccupazione", note: "MAE rolling 0.80 pp", color: C.teal });
  kpi(slide, ctx, { x: 365, y: 185, w: 260, value: "lag-1", label: "migliore per occupati", note: "MAE rolling 15.4k", color: C.blue });
  kpi(slide, ctx, { x: 655, y: 185, w: 260, value: "Logit", label: "classificazione criticità", note: "F1 0.92; ROC AUC 0.97", color: C.coral });
  ctx.addText(slide, { text: "Takeaway finale", x: 90, y: 365, w: 250, h: 34, size: 22, color: C.navy, bold: true, typeface: ctx.fonts.title });
  ctx.addText(slide, { text: "La disoccupazione italiana è persistente, territoriale e legata a fragilità sociali come il NEET. Il PIL assoluto spiega la scala economica, ma il PIL per occupato aiuta a capire la capacità del territorio di generare lavoro stabile.", x: 90, y: 410, w: 980, h: 95, size: 21, color: C.ink });
  ctx.addShape(slide, { x: 88, y: 542, w: 1010, h: 76, fill: C.navy, line: ctx.line() });
  ctx.addText(slide, { text: "Raccomandazione: concentrare politiche attive, formazione tecnica e investimenti produttivi dove disoccupazione prevista, alto NEET e basso PIL per occupato si rafforzano.", x: 115, y: 560, w: 950, h: 42, size: 18, color: "#FFFFFF", bold: true, typeface: ctx.fonts.title });
  `),
  slideModule(16, `
  base(slide, ctx, "Next steps");
  title(slide, ctx, "Prossimi sviluppi da implementare", "Il progetto puo evolvere da analisi predittiva a strumento operativo per monitorare e simulare politiche del lavoro.");
  const steps = [
    ["1", "Aggiornare e ampliare i dati", "Integrare nuovi anni ISTAT, dati provinciali, settori produttivi, istruzione, imprese attive e investimenti pubblici."],
    ["2", "Migliorare i modelli", "Testare modelli spazio-temporali e boosting avanzato, mantenendo sempre il confronto con baseline semplici e OLS."],
    ["3", "Spiegare meglio le previsioni", "Usare SHAP o feature importance locali per capire perche una regione viene classificata come critica."],
    ["4", "Costruire scenari what-if", "Simulare l'effetto di riduzioni dei NEET, aumento del PIL per occupato o nuove politiche attive sul rischio regionale."],
    ["5", "Dashboard interattiva", "Pubblicare mappe e ranking regionali aggiornabili, utili per presentare le priorita di intervento in modo immediato."]
  ];
  steps.forEach((s, i) => {
    const x = i < 3 ? 78 : 668;
    const y = i < 3 ? 172 + i * 128 : 218 + (i - 3) * 150;
    const color = [C.teal, C.blue, C.coral, C.green, C.violet][i];
    ctx.addShape(slide, { x, y, w: 58, h: 58, fill: color, line: ctx.line() });
    ctx.addText(slide, { text: s[0], x, y: y + 9, w: 58, h: 38, size: 28, color: "#FFFFFF", bold: true, align: "center", typeface: ctx.fonts.title });
    ctx.addText(slide, { text: s[1], x: x + 78, y: y - 2, w: 390, h: 28, size: 20, color: C.navy, bold: true, typeface: ctx.fonts.title });
    ctx.addText(slide, { text: s[2], x: x + 78, y: y + 32, w: 425, h: 62, size: 14, color: C.ink });
  });
  claim(slide, ctx, "Il passo successivo e trasformare il modello in un sistema di monitoraggio: aggiornabile, spiegabile e orientato alle decisioni.");
  `),
  slideModule(17, `
  ctx.addShape(slide, { x: 0, y: 0, w: ctx.W, h: ctx.H, fill: C.navy, line: ctx.line() });
  ctx.addShape(slide, { x: 0, y: 0, w: ctx.W, h: ctx.H, fill: "linear(10deg, #16213E 0%, #2A9D8F 100%)", line: ctx.line() });
  ctx.addText(slide, { text: "Grazie per l'attenzione", x: 95, y: 105, w: 900, h: 82, size: 48, color: "#FFFFFF", bold: true, typeface: ctx.fonts.title });
  ctx.addText(slide, { text: "Previsione di occupazione e disoccupazione regionale in Italia", x: 99, y: 198, w: 880, h: 36, size: 22, color: "#E6F4F1" });
  ctx.addShape(slide, { x: 92, y: 315, w: 900, h: 156, fill: "#FFFFFF", line: { style: "solid", fill: "#D8DDD2", width: 1 } });
  ctx.addShape(slide, { x: 92, y: 315, w: 10, h: 156, fill: C.gold, line: ctx.line() });
  ctx.addText(slide, { text: "Repository GitHub del progetto", x: 130, y: 342, w: 520, h: 30, size: 23, color: C.navy, bold: true, typeface: ctx.fonts.title });
  ctx.addText(slide, { text: "https://github.com/SimoneMantero/big-data-prog", x: 130, y: 386, w: 760, h: 32, size: 22, color: C.blue, bold: true });
  ctx.addText(slide, { text: "Per scaricare il progetto:", x: 130, y: 452, w: 280, h: 24, size: 15, color: C.navy, bold: true });
  ctx.addShape(slide, { x: 130, y: 492, w: 820, h: 54, fill: "#0F172A", line: ctx.line() });
  ctx.addText(slide, { text: "git clone https://github.com/SimoneMantero/big-data-prog.git", x: 153, y: 509, w: 760, h: 26, size: 18, color: "#FFFFFF" });
  ctx.addText(slide, { text: "Notebook, dati, funzioni Python, grafici, metriche e PowerPoint finale sono inclusi nella repository.", x: 96, y: 610, w: 900, h: 34, size: 18, color: "#E6F4F1" });
  ctx.addText(slide, { text: "Simone Mantero", x: 980, y: 620, w: 210, h: 28, size: 18, color: "#FFFFFF", bold: true, align: "right" });
  `),
];

async function main() {
  await fs.rm(workspace, { recursive: true, force: true });
  await fs.mkdir(slidesDir, { recursive: true });
  await fs.mkdir(previewDir, { recursive: true });
  await fs.mkdir(layoutDir, { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });
  await fs.mkdir(outputDir, { recursive: true });

  await write(path.join(workspace, "profile-plan.txt"), [
    "task mode: create",
    "primary deck-profile: strategy-leadership",
    "secondary gates: analytics narrative, policy recommendations, exact model metrics",
    "proof objects: EDA charts, model metrics, backtesting, feature importance, policy ranking, critical-region classification",
    "source requirements: local notebooks/CSV outputs and generated figures from the project workspace",
    "known missing inputs: no institutional template or logo supplied; use neutral academic design",
    "",
  ].join("\\n"));
  await write(path.join(workspace, "source-notes.txt"), [
    "Sources used:",
    "- data/processed/forecast_metrics.csv",
    "- data/processed/rolling_backtest_summary.csv",
    "- data/processed/policy_priority_table_2025.csv",
    "- data/processed/critical_region_classification_metrics.csv",
    "- reports/figures/*.png generated by project notebooks",
    "",
  ].join("\\n"));
  await write(path.join(slidesDir, "common.mjs"), common);
  for (let i = 0; i < slides.length; i += 1) {
    await write(path.join(slidesDir, `slide-${String(i + 1).padStart(2, "0")}.mjs`), slides[i]);
  }

  const buildScript = path.join(skillDir, "scripts/build_artifact_deck.mjs");
  const result = spawnSync(nodeBin, [
    buildScript,
    "--workspace", workspace,
    "--slides-dir", slidesDir,
    "--out", finalPptx,
    "--preview-dir", previewDir,
    "--layout-dir", layoutDir,
    "--contact-sheet", path.join(previewDir, "contact-sheet.png"),
    "--slide-count", "17",
  ], {
    cwd: projectRoot,
    encoding: "utf8",
    env: { ...process.env, PYTHON: "/Users/simon/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3" },
  });
  if (result.status !== 0) {
    console.error(result.stdout);
    console.error(result.stderr);
    process.exit(result.status ?? 1);
  }
  console.log(result.stdout);

  await write(path.join(qaDir, "comeback-scorecard.txt"), [
    "story: 4",
    "specificity: 5",
    "rhythm: 4",
    "whitespace: 4",
    "chart clarity: 4",
    "typography: 4",
    "restraint: 4",
    "precision: 5",
    "coherence: 4",
    "reference delta: n/a",
    "total: 38/45 before visual review; rerender contact sheet and inspect full-size slides.",
    "",
  ].join("\\n"));

  console.log(JSON.stringify({ finalPptx, workspace, contactSheet: path.join(previewDir, "contact-sheet.png") }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
