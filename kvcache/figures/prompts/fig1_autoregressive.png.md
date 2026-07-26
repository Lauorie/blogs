Educational diagram of ONE autoregressive generation step in a large language model.
Layout, left to right:
1. A vertical column of 4 stacked light-blue token boxes containing the words "The", "cat", "sat", "on" (the current input sequence).
2. A dashed arrow into one tall rounded rectangle labeled "Transformer layers".
3. Output: a vertical column of 4 gray hidden-state boxes labeled "h1", "h2", "h3", "h4". Boxes h1, h2, h3 are drawn faded (40% opacity) with a small shared label "computed, then never used". Box h4 has a bold orange border and label "the only one we need".
4. From h4 only: arrow to a small box "LM head", then to a tiny bar chart labeled "logits", then to a light-blue box with the sampled word "the".
5. A long dashed arrow from the sampled word "the" curving back to the bottom of the input column on the far left, labeled "append and repeat".
Bottom caption in dark gray: "To pick the next token, only the LAST hidden state matters".

STYLE (shared across a 5-figure series - follow exactly): Clean flat vector illustration for a technical blog, 16:9 landscape, pure white background, thin dark-gray rounded outlines, sans-serif labels, generous whitespace. Consistent palette: token/text boxes = light blue fill with blue border; Query vectors = warm orange; Key vectors = teal; Value vectors = light purple; cache/storage = light green fill with green border; wasted/redundant compute = light red fill with red border and diagonal hatching. Dashed dark-gray arrows for data flow. All text in ENGLISH only, as few words as possible, every word correctly spelled. No 3D, no gradients, no photos, no watermark, no decorative clutter.
