import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.geometry.*;
import javafx.scene.text.Font;
import javafx.stage.*;
import java.io.*;
import java.util.*;
import com.google.gson.*;

public class ChordSongEditor {

    private static final File PROJECT_FOLDER = new File("C:\\Users\\Trevor\\Dev\\Guitar Tab\\Songs\\Chords");

    private Stage primaryStage;
    private Runnable homeCallback;
    private ChordSong currentChordSong = new ChordSong();

    public ChordSongEditor(Stage primaryStage, Runnable homeCallback) {
        this.primaryStage = primaryStage;
        this.homeCallback = homeCallback;
    }

    class Chord {
        String text;
        double x;
        int lineIndex;

        Chord(String text, double x, int lineIndex) {
            this.text = text;
            this.x = x;
            this.lineIndex = lineIndex;
        }
    }

    class ChordSong {
        String title;
        String lyrics;
        ArrayList<Chord> chords = new ArrayList<>();
    }

    // Chord Editor
    // Paste lyrics
    public void openChordEditor() {

        TextField titleField = new TextField("Enter song title");

        TextArea lyricsInput = new TextArea();
        lyricsInput.setPromptText("Paste lyrics here...");
        lyricsInput.setWrapText(true);

        Button nextBtn = new Button("Next");
        Button homeBtn = new Button("Home");

        homeBtn.setOnAction(e -> homeCallback.run());

        nextBtn.setOnAction(e -> {
        currentChordSong = new ChordSong();
        currentChordSong.title = titleField.getText();
        currentChordSong.lyrics = lyricsInput.getText();
        openChordDisplay(currentChordSong);
        });

        VBox layout = new VBox(15,homeBtn, new Label("Song Title"), titleField, new Label("Paste Lyrics"), lyricsInput, nextBtn);

        layout.setPadding(new Insets(20));

        primaryStage.setScene(new Scene(layout, 600, 500));
    }

    // Display lyrics with chord input
    public void openChordDisplay(ChordSong song) {

        VBox chordDisplay = new VBox(15);
        chordDisplay.setPadding(new Insets(20));

        ScrollPane scroll = new ScrollPane(chordDisplay);

        Button homeBtn = new Button("Home");
        Button editLyricsBtn = new Button("Change Lyrics");
        Button saveBtn = new Button("Save");

        homeBtn.setOnAction(e -> homeCallback.run());

        editLyricsBtn.setOnAction(e -> openChordEditor());

        saveBtn.setOnAction(e -> saveChordSong(song));

        String[] lines = song.lyrics.split("\n");

        for (int i = 0; i < lines.length; i++) {
            String line = lines[i];

            final int lineIndex = i;

            VBox lineBox = new VBox(5);

            // chordPane is the popup when you click above a lyric to add a chord
            Pane chordPane = new Pane();
            chordPane.setMinHeight(25);

            Label lyricLabel = new Label(line);
            lyricLabel.setFont(new Font(16));

            chordPane.setOnMouseClicked(ev -> {
                TextInputDialog dialog = new TextInputDialog();
                dialog.setHeaderText("Enter Chord");

                dialog.showAndWait().ifPresent(chord -> {
                    Chord newChord = new Chord(chord, ev.getX(), lineIndex);
                    Label chordLabel = createChordLabel(newChord, song);

                    chordLabel.setLayoutX(newChord.x);
                    chordLabel.setLayoutY(0);

                    chordPane.getChildren().add(chordLabel);
                    song.chords.add(newChord);
                });
            });

            lineBox.getChildren().addAll(chordPane, lyricLabel);
            chordDisplay.getChildren().add(lineBox);
        }
        for (Chord c : song.chords) {
            VBox lineBox = (VBox) chordDisplay.getChildren().get(c.lineIndex);
            Pane chordPane = (Pane) lineBox.getChildren().get(0);

            Label chordLabel = createChordLabel(c, song);
            chordLabel.setLayoutX(c.x);
            chordLabel.setLayoutY(0);

            chordPane.getChildren().add(chordLabel);
        }

        HBox bottomBar = new HBox(10, homeBtn, editLyricsBtn, saveBtn);
        bottomBar.setPadding(new Insets(10));

        BorderPane root = new BorderPane();
        root.setCenter(scroll);
        root.setBottom(bottomBar);

        primaryStage.setScene(new Scene(root, 900, 600));
    }

    private Label createChordLabel(Chord chord, ChordSong song) {
        Label chordLabel = new Label(chord.text);
        chordLabel.setStyle(
                "-fx-background-color: lightgray;" +
                "-fx-padding: 3;" +
                "-fx-background-radius: 5;" +
                "-fx-font-weight: bold;"
        );

        chordLabel.setOnMouseClicked(ev -> {
            TextInputDialog dialog = new TextInputDialog(chord.text);
            dialog.setHeaderText("Edit Chord");
            dialog.setContentText("Chord:");

            dialog.showAndWait().ifPresent(newChord -> {
                String editedChord = newChord.trim();

                if (editedChord.isEmpty()) {
                    song.chords.remove(chord);
                    Pane parentPane = (Pane) chordLabel.getParent();
                    parentPane.getChildren().remove(chordLabel);
                    return;
                }

                chord.text = editedChord;
                chordLabel.setText(editedChord);
            });

            ev.consume();
        });

        return chordLabel;
    }

    // Save and load
    private void saveChordSong(ChordSong song) {
        FileChooser fc = new FileChooser();
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("JSON", "*.json"));
        fc.setInitialDirectory(PROJECT_FOLDER);
        fc.setInitialFileName(song.title + ".json");
        File file = fc.showSaveDialog(primaryStage);
        
        if (file == null) return;

        Gson gson = new Gson();
        try (FileWriter writer = new FileWriter(file)) {
            gson.toJson(song, writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadChordSong() {
        FileChooser fc = new FileChooser();
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("JSON", "*.json"));
        fc.setInitialDirectory(PROJECT_FOLDER);
        File file = fc.showOpenDialog(primaryStage);

        if (file == null) return;

        try {
            String json = new String(java.nio.file.Files.readAllBytes(file.toPath()));
            Gson gson = new Gson();
            ChordSong song = gson.fromJson(json, ChordSong.class);

            openChordDisplay(song);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
