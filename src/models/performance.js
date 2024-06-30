// src/models/performance.js

import mongoose from 'mongoose';

const { Schema } = mongoose;

const performanceSchema = new Schema({
  userId: { type: Schema.Types.ObjectId, ref: 'User', required: true }, // Reference to the user
  quizId: { type: Schema.Types.ObjectId, ref: 'Quiz', required: true }, // Reference to the quiz
  score: { type: Number, required: true }, // Score achieved in the quiz
  // Other fields as needed (date, feedback, etc.)
});

const Performance = mongoose.model('Performance', performanceSchema);

export default Performance;
