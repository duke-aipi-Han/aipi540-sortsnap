# SortSnap Data

Not committed to GitHub.

## Candidate Datasets

Good starting points:

- Garbage Classification dataset with classes such as cardboard, glass, metal, paper, plastic, and trash.
- Recyclable and Household Waste Classification dataset.
- Waste Classification Data with organic/recyclable classes as a fallback.

Document the exact dataset URL, license, and download date when you choose the final dataset.

## Expected Folder Structure

Use a torchvision `ImageFolder` layout:

```text
data/
  raw/
  processed/
    train/
      cardboard/
      paper/
      plastic/
      metal/
      glass/
      trash/
    val/
      cardboard/
      paper/
      plastic/
      metal/
      glass/
      trash/
    test/
      cardboard/
      paper/
      plastic/
      metal/
      glass/
      trash/
```

## Initial Class Mapping

Preferred labels:

- `cardboard`
- `paper`
- `plastic`
- `metal`
- `glass`
- `food_waste` or `organic`
- `plastic_wrap`, if available
- `other_trash` or `trash`

If the selected dataset does not include `plastic_wrap` or `food_waste`, train with the available classes and document that limitation in this file and the README.
