// src/models/note.js

import mongoose from 'mongoose';

const { Schema } = mongoose;

const noteSchema = new Schema({
  topic: { type: String, required: true },
  content: { type: String, required: true },
  // Other fields as needed (createdBy, createdAt, tags, etc.)
});

const Note = mongoose.model('Note', noteSchema);

export default Note;
