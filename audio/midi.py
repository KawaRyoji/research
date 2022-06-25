num_notes = 128
num_instruments = 128

def notes2Hz(note_number: int):
    assert(note_number >= 0 and note_number < num_notes)
    
    return 440 * 2 ** ((note_number - 69) / 12)

def is_same_chroma(note_a, note_b):
    assert(note_a >= 0 and note_a < num_notes)
    assert(note_b >= 0 and note_b < num_notes)
    
    return note_a % 12 == note_b % 12
        