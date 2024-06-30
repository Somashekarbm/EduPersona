import axios from 'axios';
import mongoose from 'mongoose';
import config from './config.js';
import Front from './src/models/basicfrontend.js';

const { MONGO_URI } = config;

const url = 'https://www.w3schools.com/python/';

// Connect to MongoDB
mongoose.connect(MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});
const db = mongoose.connection;
db.once('open', async () => {
  console.log('Connected to MongoDB');

  try {
    const response = await axios.get(url);
    const html = response.data;

    // Clean up HTML content for storage
    const cleanedHtml = html.replace(/\n\s+/g, ''); // Remove new lines and extra spaces

    // Save the content to MongoDB
    const newContent = new Front({ content: cleanedHtml });
    const savedContent = await newContent.save();
    console.log('Content saved:', savedContent);
  } catch (error) {
    console.error('Error saving content:', error);
  } finally {
    mongoose.connection.close(); // Close the connection after saving or error
  }
});
