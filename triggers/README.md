# Trigger Patterns

This directory contains the trigger patterns used in W-TAP.

## ISO 7010 Warning Symbols

W-TAP leverages ISO 7010 standardized warning symbols as trigger patterns. These symbols include:

- **W001**: Warning - General danger
- **W002**: Warning - Explosive material
- **W003**: Warning - Flammable material
- **W004**: Warning - Oxidizing material
- **W005**: Warning - Compressed gas
- **W006**: Warning - Corrosive material
- **W007**: Warning - Toxic material
- **W008**: Warning - Harmful to health
- **W009**: Warning - Environmental hazard
- **W010**: Warning - Ionizing radiation

## Obtaining Trigger Images

ISO 7010 symbols can be obtained from:

1. **ISO Official Website**: https://www.iso.org/standard/54554.html
2. **Public Domain Sources**: Various public domain symbol libraries
3. **Custom Generation**: Generate using vector graphics software

## Creating Custom Triggers

To create custom trigger patterns:

1. Use PNG format with transparent background
2. Maintain square aspect ratio
3. Ensure high contrast design
4. Size should be at least 128x128 pixels

## Usage

Place trigger images in this directory and reference them in the dataset preparation script:

```bash
python data/prepare_dataset.py \
    --trigger_image ./triggers/W001.png \
    ...
```
