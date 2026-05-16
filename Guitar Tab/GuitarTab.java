import javafx.scene.Scene;
import javafx.scene.canvas.*;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;
import javafx.stage.*;
import java.io.*;
import java.util.*;
import com.google.gson.*;

public class GuitarTab {

    private static final File PROJECT_FOLDER = new File("C:\\Users\\Trevor\\Dev\\Guitar Tab\\Songs\\Tabs");

    // Constants
    private final int WIDTH = 875;
    private final int HEIGHT = 2000;
    private final int LINE_SPACING = 20;
    private final int TAB_HEIGHT = 170;

    // Canvas acts as the paper
    private Canvas canvas;
    // GraphicsContext is what we use to draw on the canvas
    private GraphicsContext gc;

    private ArrayList<Tab> tabs = new ArrayList<>();

    private double hoverX = -1;
    private double hoverY = -1;

    private TextField titleField;

    private Stage primaryStage;
    private Runnable homeCallback;

    public GuitarTab(Stage primaryStage, Runnable homeCallback) {
        this.primaryStage = primaryStage;
        this.homeCallback = homeCallback;
    }

    // Circle/Note Class
    class Circle {
        double x, y;
        String text;

        Circle(double x, double y, String text) {
            this.x = x;
            this.y = y;
            this.text = text;
        }
    }

    // Tab Class
    class Tab {
        double offset;
        ArrayList<Circle> circles = new ArrayList<>();

        Tab(double offset) {
            this.offset = offset;
        }
    }

    public void openEditor(String jsonData) {
        openEditor(jsonData, "Enter song title");
    }

    private void openEditor(String jsonData, String songTitle) {

        tabs.clear();
        tabs.add(new Tab(0));

        canvas = new Canvas(WIDTH, HEIGHT);
        gc = canvas.getGraphicsContext2D();

        titleField = new TextField(songTitle);

        ScrollPane scrollPane = new ScrollPane(canvas);

        Button homeBtn = new Button("Home");
        Button saveBtn = new Button("Save");
        Button addTabBtn = new Button("Add Tab");

        homeBtn.setOnAction(e -> homeCallback.run());

        addTabBtn.setOnAction(e -> {
            tabs.add(new Tab(tabs.size() * TAB_HEIGHT));
            draw();
        });

        saveBtn.setOnAction(e -> saveTab());

        HBox controls = new HBox(10, homeBtn, saveBtn, addTabBtn);

        VBox top = new VBox(titleField, controls);

        BorderPane root = new BorderPane();
        root.setTop(top);
        root.setCenter(scrollPane);

        setupMouse();

        if (jsonData != null) loadFromJson(jsonData);

        draw();

        primaryStage.setScene(new Scene(root, 900, 600));
    }

    // Mouse Handling
    private void setupMouse() {
        canvas.setOnMouseMoved(e -> {
            double y = getClosestLine(e.getY());

            if (y != -1) {
                hoverX = e.getX();
                hoverY = y;
            } else hoverX = -1;

            draw();
        });

        canvas.setOnMouseClicked(e -> {
            double y = getClosestLine(e.getY());
            if (y == -1) return;

            Tab tab = getTab(y);

            Circle existing = findCircle(e.getX(), y);

            if (existing != null) {
                showMenu(e.getX(), e.getY(), existing);
                return;
            }

            showMenu(e.getX(), y, null);
        });
    }

    // Context Menu
    // Options when you add a note or click an existing one
    private void showMenu(double x, double y, Circle circle) {
        ContextMenu menu = new ContextMenu();

        // Delete option
        MenuItem delete = new MenuItem("Delete");
        delete.setOnAction(e -> {
            if (circle != null) {
                getTab(circle.y).circles.remove(circle);
                draw();
            }
        });
        menu.getItems().add(delete);

        // Fret options
        for (int i = 0; i <= 20; i++) {
            int val = i;
            MenuItem item = new MenuItem(String.valueOf(i));
            item.setOnAction(e -> {
                if (circle != null) {
                    circle.text = String.valueOf(val);
                } else {
                    getTab(y).circles.add(new Circle(x, y, String.valueOf(val)));
                }
                draw();
            });
            menu.getItems().add(item);
        }

        menu.show(canvas, x + canvas.getScene().getWindow().getX(),
                y + canvas.getScene().getWindow().getY());
    }

    // Draw
    private void draw() {
        gc.clearRect(0, 0, WIDTH, HEIGHT);

        for (Tab tab : tabs) drawTab(tab);

        drawHover();
    }

    // Draws the lines and notes for a single tab
    private void drawTab(Tab tab) {
        double left = 60;
        double right = WIDTH + 50;

        String[] labels = {"E", "A", "D", "G", "B", "E"};

        gc.setFill(Color.BLACK);
        gc.setStroke(Color.BLACK);
        gc.setFont(new Font(12));
        gc.setTextAlign(TextAlignment.LEFT);

        for (int i = 0; i < 6; i++) {
            double y = tab.offset + 60 + i * LINE_SPACING;

            gc.strokeLine(left, y, right, y);
            gc.fillText(labels[i], 30, y + 5);
        }

        gc.setTextAlign(TextAlignment.CENTER);
        gc.setFont(new Font(12));
        for (Circle c : tab.circles) {
            gc.setFill(Color.WHITE);
            gc.fillOval(c.x - 10, c.y - 10, 20, 20);

            gc.setStroke(Color.BLACK);
            gc.strokeOval(c.x - 10, c.y - 10, 20, 20);

            gc.setFill(Color.BLACK);
            gc.setFont(new Font(12));
            gc.fillText(c.text, c.x, c.y + 4);
        }
    }

    // Draws the hover circle when you move the mouse around
    private void drawHover() {
        if (hoverX == -1) return;
        if (findCircle(hoverX, hoverY) != null) return;

        double left = 60;
        double right = WIDTH + 50;
        if (hoverX < left || hoverX > right) return;

        gc.setFill(Color.rgb(128, 128, 128, 0.35));
        gc.fillOval(hoverX - 9, hoverY - 9, 18, 18);

        gc.setStroke(Color.GRAY);
        gc.strokeOval(hoverX - 9, hoverY - 9, 18, 18);
    }

    // Helpers
    // Always keeps the notes in the middle of the closest string line
    private double getClosestLine(double y) {
        double closest = -1, min = Double.MAX_VALUE;

        for (Tab t : tabs) {
            for (int i = 0; i < 6; i++) {
                double ly = t.offset + 60 + i * LINE_SPACING;
                double d = Math.abs(y - ly);
                if (d < min) {
                    min = d;
                    closest = ly;
                }
            }
        }
        return min < LINE_SPACING ? closest : -1;
    }

    // Gets the tab based on the y coordinate of the mouse
    private Tab getTab(double y) {
        for (Tab t : tabs)
            if (y >= t.offset && y <= t.offset + TAB_HEIGHT)
                return t;
        return tabs.get(0);
    }

    // When you click on an existing note, you can edit/delete it rather than adding a new one
    private Circle findCircle(double x, double y) {
        for (Tab t : tabs)
            for (Circle c : t.circles)
                if (Math.abs(c.x - x) < 10 && Math.abs(c.y - y) < 10)
                    return c;
        return null;
    }

    // Save and Load
    private void saveTab() {
        FileChooser fc = new FileChooser();
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("JSON", "*.json"));
        fc.setInitialDirectory(PROJECT_FOLDER);
        fc.setInitialFileName(titleField.getText() + ".json");
        File file = fc.showSaveDialog(primaryStage);

        if (file == null) return;

        Gson gson = new Gson();
        try (FileWriter writer = new FileWriter(file)) {
            gson.toJson(tabs, writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadTab() {
        FileChooser fc = new FileChooser();
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("JSON", "*.json"));
        fc.setInitialDirectory(PROJECT_FOLDER);
        File file = fc.showOpenDialog(primaryStage);

        if (file == null) return;

        // Load the JSON data and open the editor with it
        try {
            String json = new String(java.nio.file.Files.readAllBytes(file.toPath()));
            openEditor(json, getFileNameWithoutExtension(file));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Helper to get the song name when you load a tab
    private String getFileNameWithoutExtension(File file) {
        String fileName = file.getName();
        int dotIndex = fileName.lastIndexOf('.');

        if (dotIndex <= 0) return fileName;

        return fileName.substring(0, dotIndex);
    }

    private void loadFromJson(String json) {
        Gson gson = new Gson();
        Tab[] loaded = gson.fromJson(json, Tab[].class);

        tabs.clear();
        tabs.addAll(Arrays.asList(loaded));
    }
}
