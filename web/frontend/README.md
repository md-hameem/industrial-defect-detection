# DefectAI Frontend

Next.js 15 frontend for Industrial Defect Detection web application.

## Features

- 🏠 **Homepage** - Animated hero, features grid, AI models showcase
- 🔍 **Detection Page** - Upload images, run models, view results
- 📊 **Research Page** - Interactive performance tables, thesis figures
- 📜 **History Page** - Track past predictions with filters
- ℹ️ **About Page** - Project info, methodology, author details
- 🌓 **Dark/Light Mode** - Global theme support with ThemeContext
- ✨ **Animations** - Framer Motion throughout
- 📱 **Responsive** - Mobile-friendly design

## Tech Stack

- **Next.js 15** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Animations
- **Lucide React** - Icons
- **React Dropzone** - File uploads

## Setup

```bash
cd web/frontend
npm install
```

## Run

```bash
npm run dev
```

Open http://localhost:3000

## Project Structure

```
src/
├── app/
│   ├── page.tsx          # Homepage
│   ├── detect/page.tsx   # Detection interface
│   ├── research/page.tsx # Research results
│   ├── history/page.tsx  # Prediction history
│   ├── about/page.tsx    # About project
│   ├── layout.tsx        # Root layout
│   └── globals.css       # Global styles
├── components/
│   ├── ClientLayout.tsx  # Theme-aware layout wrapper
│   ├── Navbar.tsx        # Navigation with theme toggle
│   └── Footer.tsx        # Footer with links
└── context/
    └── ThemeContext.tsx  # Global dark/light mode state
```

## Environment Variables

Create `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Pages Overview

### Homepage (`/`)
- Animated hero with parallax effect
- Stats cards (15 categories, 99% CNN accuracy, etc.)
- How it works workflow
- Feature cards
- AI model showcase

### Detection (`/detect`)
- **Mode Toggle**: Autoencoders vs CNN
- **Autoencoder Settings**: Model selection (CAE/VAE/DAE), category dropdown
- **CNN Settings**: NEU 6-class classifier info
- **Upload**: Drag & drop with preview grid
- **Results**: Heatmaps, anomaly scores, class probabilities
- **Score Explanation**: Thresholds and model performance

### Research (`/research`)
- MVTec AD performance table (interactive model tabs)
- Cross-dataset evaluation table
- Thesis figures gallery with lightbox
- Model architecture comparison cards

### History (`/history`)
- Saved predictions from localStorage
- Filter by model, status (normal/suspicious/anomaly)
- Download heatmaps

### About (`/about`)
- Project goal and overview
- Key features grid
- Model architectures with AUC scores
- Methodology pipeline
- Datasets used
- Technology stack
- Author & Supervisor info

## Theme System

```tsx
import { useTheme } from "@/context/ThemeContext";

export default function MyComponent() {
  const { darkMode, toggleTheme } = useTheme();
  
  return (
    <div className={darkMode ? "bg-slate-900" : "bg-white"}>
      {/* content */}
    </div>
  );
}
```

## Build

```bash
npm run build
npm start
```

## License

MIT
