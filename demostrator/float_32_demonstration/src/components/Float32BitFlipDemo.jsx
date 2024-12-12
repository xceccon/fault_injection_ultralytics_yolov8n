import React, { useState, useEffect } from 'react';

// Styles for the component
const styles = {
  container: {
    fontFamily: 'Arial, sans-serif',
    padding: '20px',
    maxWidth: '1000px',
    margin: 'auto',
  },
  bitsRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: '10px',
  },
  bitBox: {
    width: '25px',
    height: '25px',
    border: '1px solid #333',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    margin: '2px 1px',
    cursor: 'pointer',
    userSelect: 'none',
    fontSize: '0.8em',
  },
  mutedBitBox: {
    backgroundColor: '#f2f2f2',
    borderColor: '#ccc',
  },
  flippedBitBox: {
    backgroundColor: '#fff',
  },
  activeBitBox: {
    backgroundColor: 'red',
  },
  signBit: {
    backgroundColor: 'rgba(0, 128, 0, 0.5)', // Muted green
  },
  exponentBit: {
    backgroundColor: 'rgba(144, 238, 144, 0.5)', // Muted light green
  },
  mantissaBit: {
    backgroundColor: 'rgba(173, 216, 230, 0.5)', // Muted light blue
  },
  byteSpacing: {
    marginRight: '10px', // Space between bytes
  },
  tooltipContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
    cursor: 'pointer',
    marginRight: '10px',
  },
  tooltipText: {
    visibility: 'hidden',
    width: '200px',
    backgroundColor: '#333',
    color: '#fff',
    textAlign: 'center',
    borderRadius: '5px',
    padding: '5px',
    position: 'absolute',
    zIndex: 1,
    bottom: '125%', // Position above the icon
    left: '50%',
    marginLeft: '-100px',
    opacity: 0,
    transition: 'opacity 0.3s',
  },
  tooltipTextVisible: {
    visibility: 'visible',
    opacity: 1,
  },
  instruction: {
    fontSize: '0.9em',
    marginRight: '10px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  floatValue: {
    fontSize: '1.2em',
    marginBottom: '20px',
  },
  inputContainer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
  },
  input: {
    fontSize: '1em',
    padding: '10px',
    width: '200px',
  },
  button: {
    padding: '10px 20px',
    fontSize: '1em',
    cursor: 'pointer',
  }
};

const Float32BitFlipDemo = () => {
  const [inputValue, setInputValue] = useState('');
  const [floatValue, setFloatValue] = useState(0);
  const [originalBits, setOriginalBits] = useState(Array(32).fill(0));
  const [flippedBits, setFlippedBits] = useState(Array(32).fill(0));
  const [tooltipVisible, setTooltipVisible] = useState(false);

  useEffect(() => {
    generateRandomFloat();
  }, []); // Run only on initial mount

  const generateRandomFloat = () => {
    const randomFloat = Math.random() * 2 - 1; // Random float between -1 and 1
    updateBitsFromInput(randomFloat);
  };

  const updateBitsFromInput = (num) => {
    setInputValue(num.toString());
    setFloatValue(num);
    const bits = floatToBits(num);
    setOriginalBits(bits);
    setFlippedBits([...bits]); // Initialize flipped bits to be the same as original bits
  };

  const floatToBits = (float) => {
    const buffer = new ArrayBuffer(4);
    const floatView = new Float32Array(buffer);
    const intView = new Uint32Array(buffer);
    floatView[0] = float;
    return intView[0].toString(2).padStart(32, '0').split('').map(Number);
  };

  const bitsToFloat = (bitsArray) => {
    const bitString = bitsArray.join('');
    const intVal = parseInt(bitString, 2);
    const buffer = new ArrayBuffer(4);
    const intView = new Uint32Array(buffer);
    const floatView = new Float32Array(buffer);
    intView[0] = intVal;
    return floatView[0];
  };

  const handleBitFlip = (index) => {
    const newFlippedBits = [...flippedBits];
    newFlippedBits[index] = flippedBits[index] === 1 ? 0 : 1; // Flip the bit
    setFlippedBits(newFlippedBits);
    setFloatValue(bitsToFloat(newFlippedBits));
  };

  const determineBitStyle = (index) => {
    if (index === 0) {
      return styles.signBit;
    } else if (index >= 1 && index <= 8) {
      return styles.exponentBit;
    } else {
      return styles.mantissaBit;
    }
  };

  const byteSeparator = (index) => {
    return index % 8 === 7 ? styles.byteSpacing : {};
  };

  return (
    <div style={styles.container}>
      <h2>Float32 Bit Flip Demo</h2>
      <div style={styles.inputContainer}>
        <input
          type="number"
          step="any"
          value={inputValue}
          onChange={(e) => {
            const value = e.target.value;
            if (value === '' || !isNaN(parseFloat(value))) {
              setInputValue(value);
              if (value !== '') updateBitsFromInput(parseFloat(value));
            }
          }}
          style={styles.input}
        />
        <button style={styles.button} onClick={generateRandomFloat}>Generate Random</button>
      </div>
      <div style={styles.tooltipContainer}
           onMouseEnter={() => setTooltipVisible(true)}
           onMouseLeave={() => setTooltipVisible(false)}>
        ℹ️
        <span style={{ ...styles.tooltipText, ...(tooltipVisible ? styles.tooltipTextVisible : {}) }}>
          <div style={{ color: 'green' }}>Sign</div>
          <div style={{ color: 'lightgreen' }}>Exponent</div>
          <div style={{ color: 'lightblue' }}>Mantissa</div>
        </span>
      </div>
      <div style={styles.floatValue}>Current Float Value: {floatValue.toFixed(7)}</div>
      <div style={styles.bitsRow}>
        {originalBits.map((bit, index) => (
          <div key={index} style={{ ...styles.bitBox, ...styles.mutedBitBox, ...determineBitStyle(index), ...byteSeparator(index) }}>
            {bit}
          </div>
        ))}
      </div>
      <div style={styles.bitsRow}>
        <span style={styles.instruction}>Bitflip egiteko, click egin bit baten gainean -></span>
        {flippedBits.map((bit, index) => (
          <div
            key={index}
            style={{
              ...styles.bitBox,
              ...(flippedBits[index] !== originalBits[index] ? styles.activeBitBox : styles.flippedBitBox),
              ...byteSeparator(index),
            }}
            onClick={() => handleBitFlip(index)}
          >
            {bit}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Float32BitFlipDemo;
