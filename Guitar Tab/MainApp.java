import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.layout.*;
import javafx.geometry.*;
import javafx.stage.*;
import java.io.*;

public class MainApp extends Application {

    private Stage primaryStage;

    private GuitarTab guitarTab;
    private ChordSongEditor chordSongEditor;

    @Override
    public void start(Stage stage) {
        this.primaryStage = stage;
        guitarTab = new GuitarTab(primaryStage, this::showHomeScreen);
        chordSongEditor = new ChordSongEditor(primaryStage, this::showHomeScreen);
        showHomeScreen();
    }

    // Home Screen
    private void showHomeScreen() {
        VBox layout = new VBox(30);
        layout.setAlignment(Pos.CENTER);
        layout.setPadding(new Insets(30));

        // Guitar Background
        File imageFile = new File("background.jpg");
        if (imageFile.exists()) {
            Image backgroundImage = new Image(imageFile.toURI().toString());
            BackgroundSize backgroundSize = new BackgroundSize(
                    100, 100, true, true, false, true);
            BackgroundImage bgImage = new BackgroundImage(
                    backgroundImage,
                    BackgroundRepeat.NO_REPEAT,
                    BackgroundRepeat.NO_REPEAT,
                    new BackgroundPosition(
                            Side.LEFT, 0, true,
                            Side.TOP, 50, true),
                    backgroundSize);
            layout.setBackground(new Background(bgImage));
        }

        // Button options from the home screen
        Button newBtn = new Button("New Tab");
        Button loadBtn = new Button("Load Tab");
        Button newChord = new Button("New Chord Song");
        Button loadChord = new Button("Load Chord Song");
        Button intructionsBtn = new Button("Instructions");

        newBtn.setPrefSize(220, 60);
        loadBtn.setPrefSize(220, 60);
        newChord.setPrefSize(220, 60);
        loadChord.setPrefSize(220, 60);
        intructionsBtn.setPrefSize(220, 60);

        newBtn.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");
        loadBtn.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");
        newChord.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");
        loadChord.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");
        intructionsBtn.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");

        // Button actions
        newBtn.setOnAction(e -> guitarTab.openEditor(null));
        loadBtn.setOnAction(e -> guitarTab.loadTab());
        newChord.setOnAction(e -> chordSongEditor.openChordEditor());
        loadChord.setOnAction(e -> chordSongEditor.loadChordSong());
        intructionsBtn.setOnAction(e -> showInstructions());

        layout.getChildren().addAll(newBtn, loadBtn, newChord, loadChord, intructionsBtn);

        primaryStage.setScene(new Scene(layout, 700, 600));
        primaryStage.setTitle("Guitar Tab Creator");
        primaryStage.show();
    }

    // Instructions
    private void showInstructions() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Instructions");
        alert.setHeaderText("How to Use the Guitar Tab Creator");
        
        TextArea textArea1 = new TextArea(
                "Tab Editor:\n" +
                "1. Click 'New Tab' to start a new guitar tab.\n" +
                "2. Click on the canvas to add notes. You can choose the fret number from the context menu.\n" +
                "3. To delete a note, click on it and select 'Delete' from the context menu.\n" +
                "4. Use the 'Save' button to save your tab as a JSON file, and 'Load' to open an existing tab."
        );
        textArea1.setWrapText(true);
        textArea1.setEditable(false);
        textArea1.setPrefWidth(500);
        textArea1.setPrefHeight(200);

        TextArea textArea2 = new TextArea(
                "Chord Editor:\n" +
                "1. Click 'New Chord Song' to start a new chord song.\n" +
                "2. Paste your lyrics into the text area and click 'Next'.\n" +
                "3. Click on the lyric lines to add chords. Enter the chord name in the dialog box.\n" +
                "4. Use the 'Save' button to save your chord song as a JSON file, and 'Load' to open an existing chord song."
        );
        textArea2.setWrapText(true);
        textArea2.setEditable(false);
        textArea2.setPrefWidth(500);
        textArea2.setPrefHeight(200);

        VBox content = new VBox(15, textArea1, textArea2);
        content.setPadding(new Insets(10));
        alert.getDialogPane().setContent(content);
        alert.showAndWait();
    }

    public static void main(String[] args) {
        launch();
    }
}