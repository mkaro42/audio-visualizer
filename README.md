# Natural-Language Audio EQ

Work in progress, built during an internship focused on context engineering and LLM
evaluation. The audio EQ use case is really just the test bed. What I'm actually trying to
get good at is: how do you write prompts/context that make an LLM produce reliable,
structured output, and how do you actually measure whether it's working (instead of just
reading the outputs and guessing)?

## The problem

If you tell an LLM "make it sound underwater," you can't just get back a paragraph. You
need real numbers a filter can use (frequency, filter type, gain) in the same format every
time. That's two problems: getting the model to reason correctly about the request
(context engineering), and getting it to output something a program can consume without
breaking (schema/output constraints). Then separately: how do you know if any of it is
actually right?

## What's in the system prompt (`eq_llm.py`)

Instead of trusting the model to just know audio EQ, I put the domain knowledge directly
in the prompt:

- Mappings from descriptive language to frequency ranges. "Underwater" absorbs highs fast
  and resonates in the low-mids, "vintage radio" is basically a 300-3000 Hz bandwidth
  limit, etc.
- A fixed JSON schema it has to return: `effect_name`, `bands` (each with frequency,
  filter_type, gain), `explanation`. No markdown, no extra text.
- Reference ranges for common cases: vocals, kick drum, cymbals, reducing room echo,
  reducing wind noise.

`main.py` takes whatever comes back and immediately runs it through `audio_filter.py`, so
if the model's output is wrong or malformed, it shows up right away as a broken filter.
There's no room for a vague answer.

## Evaluation (`generate_golden_dataset.py` + `eval_runner.py`)

The part I think matters more than the prompt itself: getting the model to sound right is
easy to fake yourself out on. So before scoring anything, `generate_golden_dataset.py`
defines known prompts with the exact expected filter bands (e.g. "apply a highpass filter
above 5000 hz" maps to highpass, 5000 Hz). It also runs the actual model against those
same prompts and saves what it returned, so I have both ground truth and real output side
by side.

`eval_runner.py` loads that golden set into LangSmith and scores the model with a few
metrics:

- `evaluate_filter_type`: did it pick the right filter type per band
- `evaluate_frequency_accuracy`: how close the predicted frequency is to expected (scored
  with some tolerance, not exact match)
- `evaluate_band_count`: did it return the right number of bands for single vs. multi-band
  requests
- an LLM-as-judge evaluator in LangSmith for the more subjective correctness check

Everything's wrapped in `@traceable` so I can go look at individual runs in LangSmith when
something scores badly, instead of just seeing a number.

## Running it

```
pip install -r requirements.txt
python generate_golden_dataset.py   # builds golden_dataset/ ground truth + LLM outputs
python eval_runner.py               # runs LangSmith eval against the golden dataset
python main.py                      # interactive CLI: load audio, describe the EQ you want, preview it
```

Needs `ANTHROPIC_API_KEY` and `LANGSMITH_API_KEY`/`LANGSMITH_PROJECT` in a `.env` file.

## Status / next steps

Still actively developing this through the internship. Prompt is getting iterated on, eval
metrics may change, and I want to add an actual results section once I have enough eval
runs to say something meaningful about accuracy over time.
