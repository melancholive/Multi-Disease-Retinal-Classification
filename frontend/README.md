# Multi-Disease Retinal Classification

A professional medical diagnostic web application for retinal disease classification using deep learning. Upload fundus images to receive AI-powered predictions for 8 different retinal conditions.

## Features

- **Image Upload Interface**: Drag-and-drop or click to upload fundus images
- **Real-Time Predictions**: ResNet18 deep learning model with 8 disease classifications
- **Professional UI**: Medical-grade design optimized for healthcare professionals
- **Dark Mode**: Complete dark mode support for reduced eye strain
- **Image Preprocessing**: Automatic 224×224 normalization with ImageNet statistics
- **Confidence Scores**: Top 5 predictions with probability visualization
- **Preprocessed Image Display**: View the normalized image sent to the model

## Supported Diseases

1. **Normal** - Healthy retina
2. **Diabetic Retinopathy** - Complications from diabetes
3. **Glaucoma** - Elevated intraocular pressure
4. **Cataract** - Lens opacity
5. **ARMD** - Age-related macular degeneration
6. **Hypertensive Retinopathy** - Hypertension-related damage
7. **Myopia** - Nearsightedness complications

## Technology Stack

### Frontend
- **Framework**: Next.js 16.2.2 with React 19.2.4
- **Language**: TypeScript 5
- **Styling**: CSS Modules
- **Features**: Server-side image serving, API integration

### Backend (Python)
- **ML Framework**: PyTorch 2.10.0
- **Model**: ResNet18 (pre-trained ImageNet, fine-tuned)
- **Image Processing**: PIL/Pillow 11.1.0, NumPy 2.1.3
- **Inference**: GPU-compatible (falls back to CPU)

## Project Structure

```
multi-disease-retinal-classification/
├── app/
│   ├── api/
│   │   ├── predict/
│   │   │   └── route.ts          # Main ML inference endpoint
│   │   ├── preprocessed/
│   │   │   └── [timestamp]/
│   │   │       └── route.ts      # Serves processed images
│   │   └── delete-preprocessed-image/
│   │       └── route.ts          # Cleanup endpoint
│   ├── components/
│   │   ├── ImageUploader.tsx      # Main upload/prediction UI
│   │   └── ImageUploader.module.css # Professional styling
│   ├── globals.css                # Global styles
│   ├── layout.tsx                 # Root layout
│   └── page.tsx                   # Home page
├── lib/
│   └── model.py                   # PyTorch inference script
├── models/
│   └── resnet_with_parameters.pth # Trained model checkpoint
├── .tmp/                          # Temporary preprocessed images
├── eslint.config.mjs              # ESLint configuration
├── next.config.ts                 # Next.js configuration
├── tsconfig.json                  # TypeScript configuration
├── package.json                   # Dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites
- Node.js 18+ with npm
- Python 3.8+ with PyTorch environment
- ~500MB disk space for model

### Frontend Setup

```bash
cd multi-disease-retinal-classification
npm install
```

### Python Environment Setup

```bash
# Create conda environment
conda create -n retinal-class python=3.13 pytorch::pytorch torchvision pillow numpy

# Activate environment
conda activate retinal-class

# Install NumPy if needed
pip install numpy==2.1.3
```

## Running the Application

### Start Development Server

```bash
npm run dev
```

Access at `http://localhost:3000`

### Build for Production

```bash
npm run build
npm start
```

## API Endpoints

### POST /api/predict
Uploads image and runs inference.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,..."
}
```

**Response:**
```json
{
  "predictions": [
    {
      "label": "Normal",
      "probability": 0.8234
    },
    {
      "label": "Diabetic Retinopathy",
      "probability": 0.1452
    }
  ],
  "preprocessedImageUrl": "/api/preprocessed/1712282154"
}
```

### GET /api/preprocessed/[timestamp]
Returns the 224×224 preprocessed image PNG.

**Response**: PNG image file

### POST /api/delete-preprocessed-image
Deletes temporary preprocessed image files.

**Request:**
```json
{
  "timestamp": "1712282154"
}
```

## Model Architecture

- **Base Model**: ResNet18 (ImageNet pre-trained)
- **Input Size**: 224×224 RGB images
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Output**: 8-class softmax probabilities
- **Fine-tuning**: Last layer unfrozen, Dropout(p=0.5)

## Image Processing Pipeline

1. **Upload**: User selects/drags image (JPEG, PNG, WebP)
2. **Base64 Encoding**: Converted to base64 on frontend
3. **Server Reception**: Next.js API receives base64
4. **Decoding**: Converted back to PIL Image
5. **Resizing**: Scaled to 224×224 maintaining aspect ratio
6. **Normalization**: Applied ImageNet statistics
7. **Saving**: PNG saved to `.tmp/preprocessed_[timestamp].png`
8. **Inference**: Model processes normalized tensor
9. **Display**: Both original and preprocessed images shown with predictions

## Performance

- **Model Inference**: ~100-200ms per image (GPU), ~500-1000ms (CPU)
- **Image Upload Limit**: 10MB max
- **Supported Formats**: JPEG, PNG, WebP, GIF, BMP
- **Preprocessing**: ~50ms
- **Server Response**: <2 seconds total

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Dark mode: All modern browsers

## Development

### File Upload Component
See `app/components/ImageUploader.tsx` for the main UI component. Features:
- Drag-and-drop file input
- File type/size validation
- Error handling with user-friendly messages
- Confidence score visualization
- Dark mode support

### Model Script
See `lib/model.py` for the PyTorch inference pipeline. Key functions:
- `load_model()`: Loads ResNet18 and checkpoint
- `preprocess_image()`: Base64 to normalized tensor
- `predict()`: Returns top 5 predictions

### Styling
CSS Modules in `app/components/ImageUploader.module.css`:
- Medical-grade color palette (#0f172a, #64748b, #3b82f6)
- Professional typography with letter-spacing
- Smooth transitions and hover states
- Complete dark mode support

## Troubleshooting

### Port 3000 already in use
```bash
lsof -i :3000
kill -9 <PID>
```

### Python not found
Ensure conda environment is activated: `conda activate retinal-class`

### Model file not found
Check `models/resnet_with_parameters.pth` exists and is readable.

### Out of memory on GPU
Set `CUDA_VISIBLE_DEVICES=-1` to force CPU inference.

## Future Enhancements

- [ ] Batch image processing
- [ ] GPU detection and optimization
- [ ] Model caching layer
- [ ] Improved error messages
- [ ] Request timeout handling
- [ ] Model ensemble predictions
- [ ] Export results as PDF report
- [ ] User authentication & history

## License

MIT License

## Authors

Developed for NYU Deep Learning course - Spring 2026

## Disclaimer

**This tool is for educational and research purposes only.** Medical professionals should not rely solely on this system for diagnosis. Always consult qualified healthcare providers for medical decisions.
