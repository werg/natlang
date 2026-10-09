---
description: The rectangle of a crop.
args:
  request: string
  source: SourceFacts
  note: Untrusted<string>
  problem?: string
returns: CropRectangle
---
Work out the rectangle of the picture to keep, in pixels from the top-left corner of the source frame.

1. Read the target size or region in request (and note when the request points to it).
2. Turn a named region into numbers with W = source.width and H = source.height:
   - a w x h rectangle in the center: x = (W - w) / 2 and y = (H - h) / 2;
   - the left half: x 0, y 0, width W / 2, height H; the right half: x W / 2, y 0, width W / 2, height H;
   - the top half: x 0, y 0, width W, height H / 2; the bottom half: x 0, y H / 2, width W, height H / 2;
   - a quarter uses W / 2 and H / 2 the same way.
3. Round x, y, width and height down to whole numbers.
4. Check x + width <= W and y + height <= H; reduce width and height until both hold. Do the arithmetic in eval.
5. When problem is given, it states what was wrong with the previous answer; give a rectangle that fixes it.
