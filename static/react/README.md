# AI Presence Frontend

Reveal the Truth Behind Every Frame.

A React-based web application for detecting deepfake content using machine learning. This application provides a user-friendly interface for uploading and analyzing videos to detect potential deepfake content.

## 🌟 Features

- **Modern and Responsive UI**
  - Clean and intuitive user interface
  - Mobile-first design approach
  - Dark/Light mode support
  - Accessible components

- **Video Processing**
  - Drag-and-drop video upload
  - Support for multiple video formats (MP4, AVI, MOV)
  - Video preview functionality
  - Progress tracking during upload

- **Deepfake Detection**
  - Real-time analysis using ML models
  - Multiple detection algorithms support
  - Confidence score visualization
  - Detailed analysis reports

- **User Experience**
  - Real-time feedback
  - Error handling and validation
  - Loading states and animations
  - Responsive design for all devices

## 📋 Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)
- Modern web browser with JavaScript enabled
- Minimum 4GB RAM recommended
- Stable internet connection

## 🚀 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ai-presence-detector.git
   ```

2. **Navigate to Project Directory**
   ```bash
   cd ai-presence-detector/static/react
   ```

3. **Install Dependencies**
   ```bash
   npm install
   ```

4. **Environment Setup**
   - Copy `.env.example` to `.env`
   - Update environment variables as needed

## 📁 Project Structure

```
static/react/
├── public/                 # Static files
│   ├── index.html
│   ├── favicon.ico
│   └── assets/
│       ├── images/
│       └── fonts/
│
├── src/                    # Source files
│   ├── components/         # Reusable components
│   │   ├── common/        # Shared components
│   │   ├── layout/        # Layout components
│   │   └── features/      # Feature-specific components
│   │
│   ├── pages/             # Page components
│   │   ├── Home/
│   │   ├── Analysis/
│   │   └── Results/
│   │
│   ├── services/          # API and service integrations
│   │   ├── api/
│   │   └── utils/
│   │
│   ├── hooks/             # Custom React hooks
│   ├── context/           # React context providers
│   ├── styles/            # Global styles and themes
│   ├── utils/             # Utility functions
│   ├── constants/         # Constants and configurations
│   ├── types/             # TypeScript type definitions
│   ├── App.tsx           # Root component
│   └── index.tsx         # Entry point
│
├── tests/                 # Test files
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore rules
├── package.json         # Project dependencies
├── tsconfig.json        # TypeScript configuration
└── README.md           # Project documentation
```

## 🏃‍♂️ Running the Application

1. **Development Mode**
   ```bash
   npm start
   ```
   - Opens `http://localhost:3000` in your default browser
   - Hot reloading enabled
   - Development tools and debugging

2. **Production Build**
   ```bash
   npm run build
   ```
   - Creates optimized production build in `build/` directory
   - Minified files and assets
   - Production-ready deployment package

## 🧪 Testing

1. **Unit Tests**
   ```bash
   npm run test
   ```

2. **Integration Tests**
   ```bash
   npm run test:integration
   ```

3. **End-to-End Tests**
   ```bash
   npm run test:e2e
   ```

## 📦 Available Scripts

- `npm start` - Start development server
- `npm run build` - Create production build
- `npm test` - Run unit tests
- `npm run lint` - Run linting
- `npm run format` - Format code
- `npm run type-check` - Check TypeScript types

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style Guidelines
- Follow ESLint configuration
- Use TypeScript for type safety
- Write meaningful commit messages
- Include tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- React Team for the amazing framework
- Contributors and maintainers
- Open source community

## 📞 Support

For support, email support@aipresencedetector.com or join our Slack channel.

---

Made with ❤️ by the AI Presence Team 