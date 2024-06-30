// src/models/quiz.js

import mongoose from 'mongoose';

const quizSchema = new mongoose.Schema({
  noteId: { type: mongoose.Schema.Types.ObjectId, ref: 'Note', required: true },
  questions: [{ type: String, required: true }],
  correctAnswers: [{ type: String, required: true }], // Update correctAnswers type to String array
});

const Quiz = mongoose.model('Quiz', quizSchema);

export default Quiz;
