// start with a 2x2 grid
let gridSize = 2;
let baseColor;
let oddColor
let oddRow;
let oddCol;
let circleSize;
let score = 0;
let scoreEl;

function setup() 
{
    createCanvas(1350, 650).parent('game');
    noStroke();

    scoreEl = document.getElementById('score');

    updateScore();

    pickColors();
}

function draw() 
{
  background(240);
  drawGrid();
}

function pickColors()
{
    baseColor = color(random(255), random(255), random(255));

    // make the odd color slightly different
    let colorChange = max(75 - (score + 5), 5);

    let whichColor = int(random(3));
    if (whichColor === 0)
        oddColor = color(red(baseColor) + colorChange, green(baseColor), blue(baseColor));
    else if (whichColor === 1)
        oddColor = color(red(baseColor), green(baseColor) + colorChange, blue(baseColor));
    else 
        oddColor = color(red(baseColor), green(baseColor), blue(baseColor) + colorChange);
    
    oddColor = color(
        constrain(red(oddColor), 0, 255),
        constrain(green(oddColor), 0, 255),
        constrain(blue(oddColor), 0, 255)
    );

    oddRow = floor(random(gridSize));
    oddCol = floor(random(gridSize));
}

function drawGrid()
{
    // to fit evenly in the screen
    let cellWidth = width / gridSize;
    let cellHeight = height / gridSize;

    circleSize = cellHeight * 0.8;

    for(let row = 0; row < gridSize; row++)
    {
        for(let col = 0; col < gridSize; col++)
        {
            let x = col * cellWidth + cellWidth / 2;
            let y = row * cellHeight + cellHeight / 2;

            if (row === oddRow && col === oddCol)
                fill(oddColor);
            else
                fill(baseColor);

            circle(x, y, circleSize);
        }
    }
}

function mousePressed()
{

    let cellWidth = width / gridSize;
    let cellHeight = height / gridSize;
    let clickedCol = floor(mouseX / cellWidth);
    let clickedRow = floor(mouseY / cellHeight);

    // correct choice
    if (clickedCol === oddCol && clickedRow === oddRow)
    {
        score++;
        gridSize++;
        updateScore();
        pickColors();
    }
    // incorrect choice
    else 
    {
    // wrong choice
        noLoop();
        setTimeout(() => {
            score = 0;
            gridSize = 2;
            updateScore();
            pickColors();
            loop();
        }, 600);
    }
}

function updateScore()
{
    if (scoreEl)
        scoreEl.textContent = `Score: ${score}`;
}