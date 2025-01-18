# LLM Cocktail Advisor

This project is an interactive cocktail recommendation system powered by a large language model (LLM). Users can explore a variety of cocktails, provide preferences, and receive personalized suggestions based on their input. The system features a dynamic web interface and utilizes advanced natural language processing techniques to enhance user experience.


## Project Structure

```
project-root/
│
├── main.py                # Main application script for core logic and cocktail handling
├── server.py              # API server script using FastAPI
├── cocktails_data.csv     # Input CSV file with cocktails data
├── requirements.txt       # Python dependencies
│
├── static/                # Static files for frontend
│   ├── app.js             # JavaScript logic for the client-side
│   └── styles.css         # CSS for frontend styling
│
├── templates/             # HTML templates for the frontend
│   └── index.html         # Main template for the user interface
│
└── README.md              # Project documentation
```

## Prerequisites

Before running this project, ensure you have the following installed on your system:

- Python 3.8 or later
- pip (Python package manager)
- Virtual environment support (optional but recommended)

## Installation

Follow these steps to set up and run the project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anthonyfracky/LLM-Cocktail-Advisor
   cd llm-cocktail-advisor
   ```

2. **Create a virtual environment:** (optional, but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Prepare the dataset:**
   Ensure `cocktails_data.csv` is present in the root directory. This file contains data about cocktails, their
   ingredients, and instructions.

## Running the Application

To start the application, follow these steps:

1. **Run the FastAPI server:**
   ```bash
   python server.py
   ```
   By default, the server will be accessible at `http://localhost:8000/`.

2. **Access the user interface:**
   Open your browser and navigate to `http://localhost:8000/` to interact with the cocktail recommendation system.

## Features

- **Personalized Recommendations:**
  The system uses an LLM to analyze user preferences and recommend cocktails tailored to their tastes.

- **Dynamic Preference Management:**
  Users can provide feedback, and the system dynamically updates its recommendations based on their preferences.

- **Rich User Interface:**
  An interactive web interface lets users browse recommendations and explore options easily.

## Project Components

### Backend

- **Framework:** FastAPI
- **Core Logic:** Implemented in `main.py` for handling cocktails and user preferences.
- **Vector Search:** Utilizes FAISS through LangChain to find similar cocktails.

### Frontend

- **HTML Templates:** Located in the `templates/`
- **CSS and JavaScript:** Found in the `static/`

## Future Enhancements

- Add output of cocktail images using `thumbnails` from the dataset.
- Optimize tracking of user preferences by improving data structure and storage mechanisms.
- Enable manual addition and removal of user preferences through the interface.
