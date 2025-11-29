import { z } from "zod";
import { Agent, AgentInputItem, Runner, withTrace } from "@openai/agents";

// Optional: lightweight web enrichment via Tavily if API key is present
async function fetchWebSnippets(query: string, depth: "basic" | "advanced", maxResults: number): Promise<{ block: string; urls: string[] } | null> {
  try {
    const apiKey = process.env.TAVILY_API_KEY;
    if (!apiKey || !query) return null;
    const res = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: apiKey,
        query,
        search_depth: depth,
        max_results: Math.max(1, Math.min(10, maxResults || 5)),
        include_answer: true,
        include_images: false,
        include_raw_content: depth === "advanced"
      })
    });
    if (!res.ok) return null;
    const data = await res.json();
    const results: Array<any> = data?.results || [];
    if (!results.length) return null;
    const urls: string[] = [];
    const top = results.slice(0, Math.min(3, results.length)).map((r: any, i: number) => {
      const title = r.title || `Result ${i + 1}`;
      const url = r.url || "";
      if (url) urls.push(url);
      const snippet = r.content || r.snippet || "";
      return `[${title}] ${url}\n${snippet}`;
    });
    return { block: top.join("\n---\n"), urls };
  } catch {
    return null;
  }
}

async function fetchTopPages(urls: string[], maxPages = 2, maxChars = 10000): Promise<string> {
  const sel = urls.slice(0, maxPages);
  const chunks: string[] = [];
  for (const url of sel) {
    try {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 7000);
      const res = await fetch(url, { signal: controller.signal, headers: { "User-Agent": "Warfare-Agent/1.0" } });
      clearTimeout(t);
      if (!res.ok) continue;
      const html = await res.text();
      const text = html
        .replace(/<script[\s\S]*?<\/script>/gi, " ")
        .replace(/<style[\s\S]*?<\/style>/gi, " ")
        .replace(/<[^>]+>/g, " ")
        .replace(/\s+/g, " ")
        .trim();
      if (text) {
        chunks.push(`URL: ${url}\nEXCERPT: ${text.slice(0, maxChars)}`);
      }
    } catch {}
  }
  return chunks.join("\n\n---\n\n");
}

function extractUserRequest(input: string): string {
  const m = input.match(/=== USER REQUEST ===\n([\s\S]*?)\n=== END USER REQUEST ===/);
  if (m) return (m[1] || "").trim().slice(0, 500);
  return input.slice(0, 500);
}

// Utility: remove any trailing "Recommendations" section to keep pure summary output
function stripRecommendations(text: string): string {
  if (!text) return text;
  // Remove from a Recommendations heading (various styles) to end of text
  const patterns = [
    /\n+#{1,3}\s*Recommendations?:[\s\S]*$/i,
    /\n+\*\*Recommendations?\*\*:[\s\S]*$/i,
    /\n+Recommendations?:[\s\S]*$/i
  ];
  for (const re of patterns) {
    if (re.test(text)) return text.replace(re, "").trim();
  }
  return text.trim();
}

// Utility: remove any trailing "Conclusion"/"Conclusions" section or common concluding phrases
function stripConclusions(text: string): string {
  if (!text) return text;
  const patterns = [
    /\n+#{1,3}\s*Conclusions?:[\s\S]*$/i,
    /\n+\*\*Conclusions?\*\*:[\s\S]*$/i,
    /\n+Conclusions?:[\s\S]*$/i,
    /\bIn\s+conclusion\b[\s\S]*$/i
  ];
  for (const re of patterns) {
    if (re.test(text)) return text.replace(re, "").trim();
  }
  return text.trim();
}

const ClassifierSchema = z.object({ classification: z.enum(["report", "decision_maker", "action_planner"]) });
const ReportClassifierSchema = z.object({ report_classification: z.enum(["overall_summary", "communications_summary"]) });
const DecisionMakerClassifierSchema = z.object({ decision_maker_classification: z.enum(["civilians", "adversaries"]) });
const classifier = new Agent({
  name: "Classifier",
  instructions: "You are a helpful conflict (warfare) assistant classifying whether a message is asking for a summary report, an action plan or decisions to take. Base your decision strictly and exclusively on the user-provided input text; do not use any external knowledge.",
  model: "gpt-4o-mini",
  outputType: ClassifierSchema,
  modelSettings: {
    store: true
  }
});

const reportClassifier = new Agent({
  name: "Report Classifier",
  instructions: "You are a helpful conflict (warfare) assistant classifying whether a message is asking for a communication summary or overall report of the situation. Base your decision strictly and exclusively on the user-provided input text; do not use any external knowledge.",
  model: "gpt-4o-mini",
  outputType: ReportClassifierSchema,
  modelSettings: {
    store: true
  }
});

const decisionMakerClassifier = new Agent({
  name: "Decision Maker Classifier",
  instructions: "You are a helpful conflict (warfare) assistant classifying whether a message is asking for decisions to take on saving eradicating the adversary or saving civilians. Base your decision strictly and exclusively on the user-provided input text; do not use any external knowledge.",
  model: "gpt-4o-mini",
  outputType: DecisionMakerClassifierSchema,
  modelSettings: {
    store: true
  }
});

const EVIDENCE_GUIDANCE = `
ACCURACY & EVIDENCE:
- Ground EVERY factual claim in the provided SOURCE DOCUMENTS where possible.
- Quote short key lines verbatim (≤160 chars) and cite with [Source:<filename>] or [Source:<heading>].
- If you add general knowledge, label it as [External] and keep it minimal.
- Never fabricate page numbers or references.

EXTREME DEPTH MANDATE:
- The user requires RIDICULOUSLY DETAILED output. Do NOT be brief.
- Expand the reasoning: objectives, assumptions, constraints, methodology (step-by-step), alternatives & trade-offs, risks & mitigations, metrics/KPIs, resources & roles, edge cases & failure modes, example scenarios, and concrete next actions.
- For each major section also add a short "Approach & Rationale" sub-section explaining WHY the items are included and HOW they would be executed.
- Expand acronyms on first mention.
- If the request is a decision brief, include evaluation criteria and scoring rationale for each COA.
- If summarizing, separate raw factual extraction from analytical synthesis.
- If planning, sequence tasks with dependencies and timing considerations.
- Target 1500–2500 words when possible.

CONFIDENCE LABELS:
- Where feasible, append (High / Medium / Low confidence) to analytical assertions. Use High only when strongly supported by multiple independent source lines.
- Highlight gaps clearly ("Information Gap:" lines) when data is insufficient.

FORMATTING:
- Use clear H2/H3 headings. Write in fully developed paragraphs as the default; use lists and tables only where they improve clarity.
- Avoid generic background unless it directly advances the user's operational understanding.
`;

const communicationsSummaryAgent = new Agent({
  name: "Communications Summary Agent",
  instructions: `${EVIDENCE_GUIDANCE}\n\nYou are a helpful conflict (warfare) assistant. Create a comprehensive report of all communications occurring in this battlefield. PRIORITIZE the provided SOURCE DOCUMENTS, but if key facts are missing, synthesize a best-effort answer using concise, relevant external knowledge and clearly mark any assumptions. Keep all content strictly relevant to the user's case; avoid generic background. Aim for ~2–3 pages (1200–1800 words). Use clear section headings, bullet points, and numbered lists where helpful. Quote key sentences verbatim and include inline citations like [Source: <filename>] or document heading titles. If you supplement with general knowledge, do not fabricate citations—label them as [External]. Do NOT include a 'Conclusion' section.`,
  model: "gpt-4o-mini",
  modelSettings: {
    store: true
  }
});

const overallSummaryAgent = new Agent({
  name: "Overall Summary Agent",
  instructions: `${EVIDENCE_GUIDANCE}\n\nYou are a helpful conflict (warfare) assistant. Produce a comprehensive situational report that flags urgent needs and threats; synthesize all information into a detailed, neutral narrative for rapid understanding by our forces. PRIORITIZE the provided SOURCE DOCUMENTS, but if key facts are missing, synthesize a best-effort answer with concise, relevant external knowledge and clearly mark any assumptions. Keep content strictly relevant to the user's case; avoid generic background. Aim for ~2–3 pages (1200–1800 words). Use section headings (e.g., Situation Overview, Key Events, Threat Assessment, Operational Constraints), bullet points, and numbered lists. Quote key sentences and include inline citations for source documents. If you supplement with general knowledge, label those lines as [External]. Do NOT include 'Recommendations' or 'Conclusion' sections.`,
  model: "gpt-4o-mini",
  modelSettings: {
    store: true
  }
});

const actionPlanAgent = new Agent({
  name: "Action Plan Agent",
  instructions: `${EVIDENCE_GUIDANCE}\n\nYou are a helpful conflict (warfare) assistant. Draft an exhaustive action plan that prioritizes saving civilians and neutralizing adversaries. PRIORITIZE the provided SOURCE DOCUMENTS, but if key inputs are missing, proceed with a best-effort plan using concise, relevant external knowledge and clearly labeled assumptions. Keep content strictly relevant to the user's case. Aim for ~2–3 pages (1200–1800 words). Include: Objectives, Assumptions, Resources, Phases/Tasks, Timelines, Risks/Mitigations, Rules of Engagement. Use bullets and numbered steps. Cite source documents inline where possible; label non-document items as [External]. Do NOT include a 'Conclusion' section.`,
  model: "gpt-4o-mini",
  modelSettings: {
    store: true
  }
});

const civiliansAgent = new Agent({
  name: "Civilians Agent",
  instructions: `${EVIDENCE_GUIDANCE}\n\nYou are a helpful conflict (warfare) assistant. Provide a detailed decision brief focused on protecting and extracting civilians at risk. PRIORITIZE the provided SOURCE DOCUMENTS, but if key facts are missing, synthesize a best-effort brief with concise, relevant external knowledge and clearly labeled assumptions. Keep content strictly relevant to the user's case. Aim for ~2–3 pages (1200–1800 words). Include Situation, Civilian Locations & Numbers, Threats to Civilians, Evacuation/Protection Plan, Force Package & Logistics, Timelines, Rules of Engagement, Risks/Mitigations. Use bullets and numbered lists. Cite source documents inline; label non-document items as [External]. Do NOT include a 'Conclusion' section.`,
  model: "gpt-4o-mini",
  modelSettings: {
    store: true
  }
});

const adversariesAgent = new Agent({
  name: "Adversaries Agent",
  instructions: `${EVIDENCE_GUIDANCE}\n\nYou are a helpful conflict (warfare) assistant. Provide a detailed decision brief on neutralizing adversaries and threats. PRIORITIZE the provided SOURCE DOCUMENTS, but if key items are missing, synthesize a best-effort brief using concise, relevant external knowledge with clearly labeled assumptions. Keep content strictly relevant to the user's case. Aim for ~2–3 pages (1200–1800 words). Include Threat Order of Battle, Disposition & Capabilities, Friendly Forces, COAs with pros/cons, Required Assets, Timelines, and Risks/Mitigations. Use bullets and numbered lists. Cite source documents inline; label non-document items as [External]. Do NOT include a 'Conclusion' section.`,
  model: "gpt-5",
  modelSettings: {
    reasoning: {
      effort: "high",
      summary: "auto"
    },
    store: true
  }
});

type WorkflowInput = { input_as_text: string; target_agent?:
  | "communications_summary"
  | "overall_summary"
  | "action_planner"
  | "decision_civilians"
  | "decision_adversaries";
  enable_web?: boolean;
  web_depth?: "basic" | "advanced";
  web_max_results?: number;
  target_word_count?: number;
  response_time_budget_ms?: number;
  verbose_mode?: boolean;
};


// Main code entrypoint
export const runWorkflow = async (workflow: WorkflowInput) => {
  return await withTrace("Replicate", async () => {
  const TIME_BUDGET_MS = Number(workflow.response_time_budget_ms || process.env.RESPONSE_TIME_BUDGET_MS || 20000);
  const concise = !workflow.verbose_mode;
    const state = {

    };
    // Accumulate a final string to return to the caller
    let final_output_text: string = "";
    // Optionally enrich with relevant web snippets if enabled
    let augmentedInput = workflow.input_as_text;
    if (workflow.target_word_count) {
      const twc = Math.min(20000, Math.max(200, workflow.target_word_count));
      augmentedInput = `${augmentedInput}\n\n(Target approximate word count: ${twc} words.)`;
    }
    if (concise) {
      augmentedInput = `IMPORTANT: Respond concisely and directly to the user request. Do NOT include methodology or meta explanations. Use short paragraphs.\n\n${augmentedInput}`;
    }
    try {
      const webEnabled = workflow.enable_web !== false; // default: enabled
      if (webEnabled) {
        const q = extractUserRequest(augmentedInput);
        const depth = workflow.web_depth || "advanced";
        const maxResults = workflow.web_max_results || 6;
        const web = await fetchWebSnippets(q, depth, maxResults);
        if (web) {
          let block = web.block;
          const pages = await fetchTopPages(web.urls, 2, 10000);
          if (pages) {
            block = `${block}\n\n=== WEB PAGE EXCERPTS ===\n${pages}`;
          }
          augmentedInput = `${augmentedInput}\n\n=== EXTERNAL WEB SNIPPETS ===\n${block}\n=== END EXTERNAL WEB SNIPPETS ===`;
        }
      }
    } catch {}

    const conversationHistory: AgentInputItem[] = [
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: augmentedInput
          }
        ]
      }
    ];
    const runner = new Runner({
      traceMetadata: {
        __trace_source__: "agent-builder",
        workflow_id: "wf_68ff06258274819088ea23ef04936e0306ec0a076c3bcec2"
      }
    });
    // If a target agent is specified, bypass classification and route directly
    if (workflow.target_agent) {
      const route = workflow.target_agent;
      let final_output_text = "";
      switch (route) {
        case "communications_summary": {
          const r = await runner.run(communicationsSummaryAgent, [...conversationHistory]);
          final_output_text = String(r.finalOutput ?? "");
          final_output_text = stripConclusions(final_output_text);
          return final_output_text;
        }
        case "overall_summary": {
          const r = await runner.run(overallSummaryAgent, [...conversationHistory]);
          final_output_text = String(r.finalOutput ?? "");
          final_output_text = stripConclusions(final_output_text);
          final_output_text = stripRecommendations(final_output_text);
          return final_output_text;
        }
        case "action_planner": {
          const r = await runner.run(actionPlanAgent, [...conversationHistory]);
          final_output_text = String(r.finalOutput ?? "");
          final_output_text = stripConclusions(final_output_text);
          return final_output_text;
        }
        case "decision_civilians": {
          const r = await runner.run(civiliansAgent, [...conversationHistory]);
          final_output_text = String(r.finalOutput ?? "");
          final_output_text = stripConclusions(final_output_text);
          return final_output_text;
        }
        case "decision_adversaries": {
          const r = await runner.run(adversariesAgent, [...conversationHistory]);
          final_output_text = String(r.finalOutput ?? "");
          final_output_text = stripConclusions(final_output_text);
          return final_output_text;
        }
        default: {
          console.warn("Unknown target_agent provided:", route, "— falling back to classification.");
        }
      }
    }

    const withTimeout = async <T>(p: Promise<T>, ms: number): Promise<T> => {
      return await Promise.race([
        p,
        new Promise<T>((_, reject) => setTimeout(() => reject(new Error("TIMEOUT_EXCEEDED")), ms))
      ]);
    };

    let classifierResultTemp;
    try {
      classifierResultTemp = await withTimeout(runner.run(
      classifier,
      [
        ...conversationHistory
      ]
      ), Math.round(TIME_BUDGET_MS * 0.25));
    } catch (e) {
      console.warn("Classifier timeout; proceeding with default route to overall_summary");
      const r = await withTimeout(runner.run(overallSummaryAgent, [...conversationHistory]), Math.round(TIME_BUDGET_MS * 0.6));
      final_output_text = String(r.finalOutput ?? "");
      final_output_text = stripConclusions(final_output_text);
      final_output_text = stripRecommendations(final_output_text);
      return final_output_text || "## Title\n\n### Situation Overview\n\n- Timeout reached.\n\n### Key Points\n\n- Partial information only.";
    }
    conversationHistory.push(...classifierResultTemp.newItems.map((item) => item.rawItem));

    // Be tolerant if an agent omits finalOutput; avoid throwing 500
    if (!classifierResultTemp.finalOutput) {
      console.warn("Classifier produced no finalOutput; proceeding with conversation history");
    }

    const classifierResult = {
      output_text: JSON.stringify(classifierResultTemp.finalOutput),
      output_parsed: classifierResultTemp.finalOutput
    };
    console.log("Classifier =>", classifierResult.output_parsed);
  if (classifierResult.output_parsed.classification == "report") {
      let reportClassifierResultTemp;
      try {
        reportClassifierResultTemp = await withTimeout(runner.run(
        reportClassifier,
        [
          ...conversationHistory
        ]
        ), Math.round(TIME_BUDGET_MS * 0.2));
      } catch (e) {
        console.warn("Report classifier timeout; defaulting to overall_summary");
        const r = await withTimeout(runner.run(overallSummaryAgent, [...conversationHistory]), Math.round(TIME_BUDGET_MS * 0.6));
        final_output_text = String(r.finalOutput ?? "");
        final_output_text = stripConclusions(final_output_text);
        final_output_text = stripRecommendations(final_output_text);
        return final_output_text;
      }
      conversationHistory.push(...reportClassifierResultTemp.newItems.map((item) => item.rawItem));

      if (!reportClassifierResultTemp.finalOutput) {
        console.warn("ReportClassifier produced no finalOutput; attempting to continue");
      }

      const reportClassifierResult = {
        output_text: JSON.stringify(reportClassifierResultTemp.finalOutput),
        output_parsed: reportClassifierResultTemp.finalOutput
      };
      console.log("ReportClassifier =>", reportClassifierResult.output_parsed);
  if (reportClassifierResult.output_parsed.report_classification == "communications_summary") {
        try {
          const communicationsSummaryAgentResultTemp = await withTimeout(runner.run(
            communicationsSummaryAgent,
            [
              ...conversationHistory
            ]
          ), Math.round(TIME_BUDGET_MS * 0.5));
          conversationHistory.push(...communicationsSummaryAgentResultTemp.newItems.map((item) => item.rawItem));

          if (!communicationsSummaryAgentResultTemp.finalOutput) {
            console.warn("CommunicationsSummary produced no finalOutput; using empty string");
          }

          final_output_text = String(communicationsSummaryAgentResultTemp.finalOutput ?? "");
          // Remove any conclusion sections or phrases
          final_output_text = stripConclusions(final_output_text);
          console.log("CommunicationsSummary =>", final_output_text.slice(0, 140));
        } catch (e) {
          console.warn("CommunicationsSummary timed out or failed:", e);
          return "Communications Summary unavailable within time budget. Providing minimal summary: Communications details require more processing—try reducing web depth or increasing time budget.";
        }
      } else if (reportClassifierResult.output_parsed.report_classification == "overall_summary") {
        try {
          const overallSummaryAgentResultTemp = await withTimeout(runner.run(
            overallSummaryAgent,
            [
              ...conversationHistory
            ]
          ), Math.round(TIME_BUDGET_MS * 0.5));
          conversationHistory.push(...overallSummaryAgentResultTemp.newItems.map((item) => item.rawItem));

          if (!overallSummaryAgentResultTemp.finalOutput) {
            console.warn("OverallSummary produced no finalOutput; using empty string");
          }

          final_output_text = String(overallSummaryAgentResultTemp.finalOutput ?? "");
          // Ensure we do not return any conclusions or recommendations for overall summary
          final_output_text = stripConclusions(final_output_text);
          final_output_text = stripRecommendations(final_output_text);
          console.log("OverallSummary (sanitized) =>", final_output_text.slice(0, 140));
        } catch (e) {
          console.warn("OverallSummary timed out or failed:", e);
          return "Overall Summary unavailable within time budget. Provide a brief prompt and retry, or increase the response time slider.";
        }
      } else {
        final_output_text = "No report summary could be produced.";
      }
      // Return immediately once a report summary decision has been made
      return final_output_text;
    } else if (classifierResult.output_parsed.classification == "decision_maker") {
      let decisionMakerClassifierResultTemp;
      try {
        decisionMakerClassifierResultTemp = await withTimeout(runner.run(
          decisionMakerClassifier,
          [
            ...conversationHistory
          ]
        ), Math.round(TIME_BUDGET_MS * 0.2));
      } catch (e) {
        console.warn("DecisionMakerClassifier timeout; returning minimal response");
        return "Decision brief unavailable within time budget. Try again with a shorter prompt or increase the time budget.";
      }
      conversationHistory.push(...decisionMakerClassifierResultTemp.newItems.map((item) => item.rawItem));

      if (!decisionMakerClassifierResultTemp.finalOutput) {
        console.warn("DecisionMakerClassifier produced no finalOutput; attempting to continue");
      }

      const decisionMakerClassifierResult = {
        output_text: JSON.stringify(decisionMakerClassifierResultTemp.finalOutput),
        output_parsed: decisionMakerClassifierResultTemp.finalOutput
      };
      console.log("DecisionMakerClassifier =>", decisionMakerClassifierResult.output_parsed);
  if (decisionMakerClassifierResult.output_parsed.decision_maker_classification == "civilians") {
        try {
          const civiliansAgentResultTemp = await withTimeout(runner.run(
            civiliansAgent,
            [
              ...conversationHistory
            ]
          ), Math.round(TIME_BUDGET_MS * 0.6));
        conversationHistory.push(...civiliansAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!civiliansAgentResultTemp.finalOutput) {
          console.warn("Civilians agent produced no finalOutput; using empty string");
        }

        final_output_text = String(civiliansAgentResultTemp.finalOutput ?? "");
        // Remove any conclusion sections or phrases
        final_output_text = stripConclusions(final_output_text);
  console.log("Civilians =>", final_output_text.slice(0, 140));
        } catch (e) {
          console.warn("Civilians agent timed out or failed:", e);
          return "Civilians decision brief unavailable within time budget. Increase the time budget or retry with fewer web results.";
        }
      } else if (decisionMakerClassifierResult.output_parsed.decision_maker_classification == "adversaries") {
        try {
          const adversariesAgentResultTemp = await withTimeout(runner.run(
            adversariesAgent,
            [
              ...conversationHistory
            ]
          ), Math.round(TIME_BUDGET_MS * 0.6));
        conversationHistory.push(...adversariesAgentResultTemp.newItems.map((item) => item.rawItem));

        if (!adversariesAgentResultTemp.finalOutput) {
          console.warn("Adversaries agent produced no finalOutput; using empty string");
        }

        final_output_text = String(adversariesAgentResultTemp.finalOutput ?? "");
        // Remove any conclusion sections or phrases
        final_output_text = stripConclusions(final_output_text);
  console.log("Adversaries =>", final_output_text.slice(0, 140));
        } catch (e) {
          console.warn("Adversaries agent timed out or failed:", e);
          return "Adversaries decision brief unavailable within time budget. Try again or increase the time budget in the UI.";
        }
      } else {
        final_output_text = "No decision-making output could be produced.";
      }
      // Return immediately after decision making
      return final_output_text;
    } else if (classifierResult.output_parsed.classification == "action_planner") {
      try {
        const actionPlanAgentResultTemp = await withTimeout(runner.run(
          actionPlanAgent,
          [
            ...conversationHistory
          ]
        ), Math.round(TIME_BUDGET_MS * 0.6));
      conversationHistory.push(...actionPlanAgentResultTemp.newItems.map((item) => item.rawItem));

      if (!actionPlanAgentResultTemp.finalOutput) {
        console.warn("ActionPlan produced no finalOutput; using empty string");
      }

      final_output_text = String(actionPlanAgentResultTemp.finalOutput ?? "");
      // Remove any conclusion sections or phrases
      final_output_text = stripConclusions(final_output_text);
      // Return immediately after action plan
      return final_output_text;
  console.log("ActionPlan =>", final_output_text.slice(0, 140));
      } catch (e) {
        console.warn("ActionPlan timed out or failed:", e);
        return "Action plan unavailable within time budget. Please increase the time budget or simplify the request.";
      }
    } else {
      final_output_text = "Unable to classify request into report, decision_maker, or action_planner.";
    }
    return final_output_text;
  });
}
