import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { runWorkflow } from "./workflow.js";

type ExpressRequest = import("express").Request;
type ExpressResponse = import("express").Response;

dotenv.config();

const app = express();
app.use(cors());
// Increase body size limits to handle large prompts/documents
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));

app.get("/health", (_req, res: ExpressResponse) => res.json({ status: "ok" }));

// Lightweight diagnostics to verify configuration without exposing secrets
app.get("/diag", (_req, res: ExpressResponse) => {
  const hasOpenAIKey = Boolean(process.env.OPENAI_API_KEY || (process as any).env?.OPENAI_KEY);
  const hasTavily = Boolean(process.env.TAVILY_API_KEY);
  res.json({
    status: "ok",
    hasOpenAIKey,
    hasTavily,
    modelDefaults: {
      report: "gpt-4o-mini",
      actionPlan: "gpt-4o-mini",
      adversaries: "gpt-5"
    }
  });
});

app.post("/run", async (req: ExpressRequest, res: ExpressResponse) => {
  try {
    const { input_as_text, target_agent, enable_web, web_depth, web_max_results, target_word_count, response_time_budget_ms, verbose_mode } = req.body ?? {};
    if (!input_as_text) return res.status(400).json({ error: "input_as_text is required" });
    // Preflight: ensure OpenAI key exists server-side to avoid silent timeouts
    const hasOpenAIKey = Boolean(process.env.OPENAI_API_KEY || (process as any).env?.OPENAI_KEY);
    if (!hasOpenAIKey) {
      return res.status(500).json({
        error: "Missing OpenAI API key on server",
        details: "Set OPENAI_API_KEY in .env or environment and restart the server."
      });
    }

    // Call your Agents SDK workflow:
  const result = await runWorkflow({ input_as_text, target_agent, enable_web, web_depth, web_max_results, target_word_count, response_time_budget_ms, verbose_mode });
    // Debug logging to trace what the workflow returned
    const resultPreview = typeof result === "string" ? result.slice(0, 140) : JSON.stringify(result).slice(0, 140);
    console.log("/run input:", input_as_text);
    console.log("/run result type:", typeof result, "preview:", resultPreview);

  // Align with Streamlit client which expects 'output_text' when present
    res.json({ output_text: result });
  } catch (err: any) {
    // Log full error for local debugging
    console.error("/run error:", err);
    res.status(500).json({
      error: err?.message || "Unknown error",
      // Surface minimal details to help diagnose locally
      details: typeof err?.stack === "string" ? err.stack.split("\n").slice(0, 3).join("\n") : undefined
    });
  }
});

const PORT = Number(process.env.PORT) || 3001;
app.listen(PORT, () => console.log(`API on http://localhost:${PORT}`));
