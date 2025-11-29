import "dotenv/config";
import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

app.post("/run", (req, res) => {
  const { input_as_text } = req.body ?? {};
  res.json({ output_text: `You sent: ${input_as_text}` });
});

const port = Number(process.env.PORT ?? 5050);
app.listen(port, () => console.log(`Listening on http://localhost:${port}`));
