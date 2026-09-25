# Contributing

Thanks for helping! Bug reports with a before/after image are the most useful contribution:
[open an issue](https://github.com/StarkAg/retype/issues/new/choose).

## Development

```bash
pip install -r requirements.txt
python3 tests/smoke_test.py          # end-to-end check on the bundled samples
python3 examples/make_samples.py     # regenerate the synthetic samples (macOS fonts)
python3 examples/make_assets.py      # regenerate assets/demo.gif and assets/social-preview.png
```

- Keep `retype.py` a single file with no dependencies beyond OpenCV, NumPy and Pillow.
- Add a case to `tests/smoke_test.py` when you fix a kind of text that used to fail.
- Only use synthetic or public-domain images in the repo. No real personal documents.
