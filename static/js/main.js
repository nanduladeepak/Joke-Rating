


// Get a reference to the form and text area elements
// const form = document.querySelector('form');
// const textarea = document.querySelector('textarea');
const textarea = document.getElementById('story');


const button = document.getElementById('submitJoke');

button.addEventListener('click',(e)=>{
    const text = textarea.value;
    // Send a POST request to the server with the text as the data
    fetch('/process_text', {
        method: 'POST',
        body: new URLSearchParams({text}),
    })
    .then(response => response.text())
    .then(result => {
        // Display the result on the web page
        const resultDiv = document.querySelector('#result');
        resultDiv.innerHTML = '';
        try {
            const res = JSON.parse(result);
            var table = document.createElement("table");
            table.id = 'jsonTable';

			// Clear the existing table
			while (table.firstChild) {
				table.removeChild(table.firstChild);
			}

			// Create the table headers
			var headerRow = document.createElement("tr");
			var headers = ['jokeLength', 'sentiment', 'sentiment_prob', 'laughingTime', 'rank'];
			var headersData = {
                'jokeLength':'Joke Word Count',
                'sentiment':'Joke Sentiment',
                'sentiment_prob':'Probability of joke sentiment',
                'laughingTime':'Predicted Audience reaction time in ms',
                'rank': 'Joke Rating out of 10'
            };
			for (var i = 0; i < headers.length; i++) {
				var headerCell = document.createElement("th");
				headerCell.textContent = headersData[headers[i]];
				headerRow.appendChild(headerCell);
			}
			table.appendChild(headerRow);

			// Create the table rows
			var row = document.createElement("tr");
			for (var i = 0; i < headers.length; i++) {
				var cell = document.createElement("td");
				cell.textContent = res[headers[i]];
				row.appendChild(cell);
			}
			table.appendChild(row);
		
            resultDiv.appendChild(table)
            // resultDiv.textContent = result;
        } catch (error) {
            resultDiv.textContent = 'Failed to get result sorry 🙁'
        }
    });
})

// Add an event listener for when the form is submitted
// form.addEventListener('submit', (event) => {
//     // Prevent the default form submission behavior
//     event.preventDefault();

//     // Get the text from the text area
//     const text = textarea.value;

//     // Send a POST request to the server with the text as the data
//     fetch('/', {
//         method: 'POST',
//         body: new URLSearchParams({text}),
//     })
//     .then(response => response.text())
//     .then(result => {
//         // Display the result on the web page
//         const resultDiv = document.querySelector('#result');
//         resultDiv.textContent = result;
//     });
// });

{/* <head>
	<title>JSON Table Example</title>
	<style>
		table {
			border-collapse: collapse;
			width: 100%;
		}
		th, td {
			text-align: left;
			padding: 8px;
			border-bottom: 1px solid #ddd;
		}
		tr:nth-child(even) {
			background-color: #f2f2f2;
		}
	</style>
</head>
<body>
	<div id="table-container"></div>
	 */}
