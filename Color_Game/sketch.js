// start with a 2x2 grid
let gridSize = 2;
let baseColor;
let oddColor;
let oddRow;
let oddCol;
let circleSize;
let score = 0;
let scoreEl;
let gameOver = false;
let finalScore = 0;

function setup() 
{
    createCanvas(windowWidth, getCanvasHeight()).parent('game');
    noStroke();

    scoreEl = document.getElementById('score');

    updateScore();

    pickColors();
}

function draw() 
{
  background(240);
  drawGrid();

  if (gameOver)
    drawGameOver();
}

function pickColors()
{
    baseColor = color(random(255), random(255), random(255));

    // make the odd color slightly different
    let colorChange = max(100 - (score + 2), 10);

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

    circleSize = min(cellWidth, cellHeight) * 0.86;

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
    if (gameOver)
    {
        restartGame();
        return;
    }

    if (mouseX < 0 || mouseX > width || mouseY < 0 || mouseY > height)
        return;

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
        finalScore = score;
        gameOver = true;
    }
}

function updateScore()
{
    if (scoreEl)
        scoreEl.textContent = `Score: ${score}`;
}

// flash a white outline around the odd circle when the game is over
function drawMismatchOutline(cellWidth, cellHeight)
{
    let flashOn = floor(frameCount / 10) % 2 === 0;
    if (!flashOn)
        return;

    let x = oddCol * cellWidth + cellWidth / 2;
    let y = oddRow * cellHeight + cellHeight / 2;

    noFill();
    stroke(255);
    strokeWeight(max(5, circleSize * 0.06));
    circle(x, y, circleSize + max(12, circleSize * 0.14));
    stroke(35, 35, 35);
    strokeWeight(max(2, circleSize * 0.025));
    circle(x, y, circleSize + max(22, circleSize * 0.22));
    noStroke();
}

// draw a game over screen with the final score
function drawGameOver()
{
    fill(0, 0, 0, 25);
    rect(0, 0, width, height);

    drawMismatchOutline(width / gridSize, height / gridSize);

    textAlign(CENTER, CENTER);
    fill(255);
    textStyle(BOLD);
    textSize(min(82, width * 0.12));
    text('GAME OVER', width / 2, height / 2 - 54);

    textStyle(NORMAL);
    textSize(min(34, width * 0.06));
    text(`Score: ${finalScore}`, width / 2, height / 2 + 22);

    textSize(min(20, width * 0.04));
    text('Click to play again', width / 2, height / 2 + 74);
}

function restartGame()
{
    score = 0;
    gridSize = 2;
    gameOver = false;
    updateScore();
    pickColors();
}

function windowResized()
{
    resizeCanvas(windowWidth, getCanvasHeight());
}

function getCanvasHeight()
{
    let header = document.querySelector('header');
    let headerHeight = header ? header.offsetHeight : 0;
    return windowHeight - headerHeight;
}
