Educational diagram of ONE decode step with a KV cache in a large language model.
Layout, left to right:
1. One light-blue token box with the word "token 7" and the label "newest token only".
2. Three small projection arrows out of it producing three small vector boxes: "q" (orange), "k" (teal), "v" (light purple).
3. The "k" and "v" boxes have solid arrows APPENDING them into a large light-green rounded container titled "KV cache (one per layer)". Inside the container: two horizontal rows of slots - top row labeled "K:" with slots "k1 k2 k3 k4 k5 k6" plus a highlighted new slot "k7" being added at the right end; bottom row labeled "V:" with slots "v1 v2 v3 v4 v5 v6" plus a highlighted new slot "v7". A small label near the new slots: "append: one k, one v per step".
4. From the whole cache container and from the "q" box, arrows converge into a rounded box labeled "attention: q x all cached K, weights x all cached V", then one arrow out to a light-blue box labeled "next token".
Bottom caption: "Compute q, k, v for ONE token. Read everything else from the cache."

STYLE (shared across a 5-figure series - follow exactly): Clean flat vector illustration for a technical blog, 16:9 landscape, pure white background, thin dark-gray rounded outlines, sans-serif labels, generous whitespace. Consistent palette: token/text boxes = light blue fill with blue border; Query vectors = warm orange; Key vectors = teal; Value vectors = light purple; cache/storage = light green fill with green border; wasted/redundant compute = light red fill with red border and diagonal hatching. Dashed dark-gray arrows for data flow. All text in ENGLISH only, as few words as possible, every word correctly spelled. No 3D, no gradients, no photos, no watermark, no decorative clutter.

CORRECTIONS (fix these exact problems from the previous render):
- The bottom caption must read EXACTLY "Compute q, k, v for ONE token. Read everything else from the cache." with the word "cache" fully and cleanly rendered - no smudged or missing letters.
- There must be exactly ONE small label inside the cache container near the new k7/v7 slots, reading exactly "append: one k, one v per step". Do NOT add any second label. Never write "q, k, one v per step" - q is never stored in the cache.
- Do NOT put any text label on the three arrows from "token 7" to the q, k, v boxes. The previous render wrote a misspelled word "projet" - remove it entirely. These arrows are unlabeled.
- The three arrows from "token 7" must point ONLY outward, from "token 7" toward q, k, v. No arrowhead may touch or point into the "token 7" box.
- Do NOT draw any arrow between the k7 slot and the v7 slot inside the cache. Nothing connects v7 to k7. Arrows toward the attention box leave from the right edge of the cache container only.

CORRECTIONS (must follow):
- Every visible word must be spelled correctly. The small label near the three projection arrows must read exactly "project" — or omit that label entirely.
- Do not draw a dashed line from the "v" box along the bottom edge to the attention box. The k and v boxes flow ONLY into the cache container; only "q" connects directly to the attention box (via the top dashed arrow).

SECOND CORRECTIONS (must follow, highest priority):
- The previous attempt corrupted some words. EVERY word must be perfectly legible, correctly spelled dictionary English.
- The bottom caption must read EXACTLY: "Compute q, k, v for ONE token. Read everything else from the cache."
- The label under the "token 7" box must read EXACTLY: "newest token only" (two words "token" in a row is forbidden).
- Remove the small "projection" label near the three arrows entirely - draw the three arrows with no text label.
