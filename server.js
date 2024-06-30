// src/server.js

import express from 'express';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import cors from 'cors'; 
import config from './config.js'; // Adjust the path based on your actual structure
import Note from './src/models/note.js';
import Quiz from './src/models/quiz.js';
import Performance from './src/models/performance.js';
import Front from './src/models/basicfrontend.js';

// Load environment variables from .env file
dotenv.config();

const app = express();
const { PORT, MONGO_URI } = config;

app.use(express.json());
app.use(cors());




// Connect to MongoDB
mongoose.connect(MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});
const db = mongoose.connection;
db.once('open', () => console.log('Connected to MongoDB'));

// Routes
app.post('/api/content', async (req, res) => {
  const { content } = req.body;

  try {
    const newContent = new Front({ content });
    await newContent.save();
    res.status(201).json(newContent);
  } catch (error) {
    console.error('Error saving content:', error);
    res.status(500).json({ message: 'Server Error' });
  }
});



app.get('/api/content', async (req, res) => {
  try {
    const contents = await Front.find();
    res.json(contents);
  } catch (error) {
    console.error('Error fetching content:', error);
    res.status(500).json({ message: 'Server Error' });
  }
});

app.post('/api/notes', async (req, res) => {
  try {
    const { topic, content } = req.body;

    const newNote = new Note({
      topic,
      content,
      // Add other fields as needed
    });

    await newNote.save();
    res.status(201).json(newNote);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server Error' });
  }
});

app.get('/api/notes', async (req, res) => {
  try {
    const notes = await Note.find();
    res.json(notes);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server Error' });
  }
});

app.post('/api/notes/:noteId/quiz', async (req, res) => {
  const { noteId } = req.params;
  const { questions, correctAnswers } = req.body;

  try {
    const quiz = new Quiz({
      noteId,
      questions,
      correctAnswers,
    });

    await quiz.save();
    res.status(201).json(quiz);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server Error' });
  }
});

app.post('/api/quiz/:quizId/performance', async (req, res) => {
  const { quizId } = req.params;
  const { userId, score } = req.body;

  try {
    const performance = new Performance({
      userId,
      quizId,
      score,
    });

    await performance.save();
    res.status(201).json(performance);
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server Error' });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
