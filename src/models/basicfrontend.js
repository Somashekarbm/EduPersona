
import mongoose from 'mongoose';

const { Schema } = mongoose;

const frontSchema = new Schema({
  content: { type: String, required: true },
    
});

const Front = mongoose.model('Front', frontSchema);
export default Front;
