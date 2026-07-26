Educational timeline diagram of the two phases of LLM inference.
A single horizontal time axis labeled "time" spanning the full width.
Phase 1 on the left, titled "PREFILL": one single WIDE and TALL light-blue block sitting on the axis, containing the text "all prompt tokens processed in parallel - builds the KV cache". Below it a small tag "compute-heavy".
Phase 2 on the right, titled "DECODE": a row of 10 small narrow light-green blocks evenly spaced on the axis, with the shared label "one token per step - reads the cache". Below them a small tag "fast, memory-bound".
Above the diagram: a horizontal bracket labeled "TTFT (time to first token)" spanning from the start of the axis to the FIRST small green block.
Under the axis: a thin light-green wedge growing from left to right labeled "KV cache size grows", getting slightly taller after each decode block.
Bottom caption: "Building the cache is expensive. Reading it is cheap."

STYLE (shared across a 5-figure series - follow exactly): Clean flat vector illustration for a technical blog, 16:9 landscape, pure white background, thin dark-gray rounded outlines, sans-serif labels, generous whitespace. Consistent palette: token/text boxes = light blue fill with blue border; Query vectors = warm orange; Key vectors = teal; Value vectors = light purple; cache/storage = light green fill with green border; wasted/redundant compute = light red fill with red border and diagonal hatching. Dashed dark-gray arrows for data flow. All text in ENGLISH only, as few words as possible, every word correctly spelled. No 3D, no gradients, no photos, no watermark, no decorative clutter.
