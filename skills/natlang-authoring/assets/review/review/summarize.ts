export default function summarize(assessments: Assessment[]): Report {
return {
  assessments: assessments,
  supported: assessments.filter(a => a.verdict === "supported").length,
  contradicted: assessments.filter(a => a.verdict === "contradicted").length,
  uncertain: assessments.filter(a => a.verdict === "uncertain").length,
};
}
