import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.TreeSet;

/**
 * Native COMSOL model: 2D surface-trap RF coupling into floating silicon.
 * Compile using COMSOL, then open the .class in COMSOL or run it in batch.
 * Geometry: 500 nm ideal metal / 12 um SiO2 / 675 um silicon.
 * Drive: 40 MHz, 100 V PEAK; silicon: 1, 2, 5, 10 ohm cm.
 * This class builds AND solves the model; it does not use an external solver.
 */
public class SiliconLeakage2D {
    private static final double[] RESISTIVITIES = {1, 2, 5, 10};

    // Transverse electrode bounds (um) from the existing surface-trap layout.
    // Each row: left edge, right edge, RF flag (1 = RF, 0 = DC/GND).
    private static final double[][] ELECTRODES = {
        {-3250, -288.5, 0},
        {-280.5, -146.5, 0},
        {-138.5,  -46.5, 1},
        { -38.5,   38.5, 0},
        {  46.5,  123.5, 1},
        { 131.5,  265.5, 0},
        { 273.5, 3250.0, 0}
    };

    private static String[] strings(List<String> values) {
        return values.toArray(new String[0]);
    }

    private static void rectangle(Model model, String tag, String x, String y,
                                  String width, String height) {
        model.component("comp1").geom("geom1").create(tag, "Rectangle");
        model.component("comp1").geom("geom1").feature(tag).set("pos", new String[]{x, y});
        model.component("comp1").geom("geom1").feature(tag).set("size", new String[]{width, height});
        model.component("comp1").geom("geom1").feature(tag).set("selresult", "on");
        model.component("comp1").geom("geom1").feature(tag).set("selresultshow", "all");
    }

    private static void union(Model model, String tag, int dimension, String[] inputs) {
        model.component("comp1").selection().create(tag, "Union");
        model.component("comp1").selection(tag).set("entitydim", dimension);
        model.component("comp1").selection(tag).set("input", inputs);
    }

    private static void boundaryBox(Model model, String tag, double xmin, double xmax,
                                    double ymin, double ymax) {
        // Selection coordinate units follow geom1.lengthUnit("um").
        model.component("comp1").selection().create(tag, "Box");
        model.component("comp1").selection(tag).set("entitydim", 1);
        model.component("comp1").selection(tag).set("xmin", xmin);
        model.component("comp1").selection(tag).set("xmax", xmax);
        model.component("comp1").selection(tag).set("ymin", ymin);
        model.component("comp1").selection(tag).set("ymax", ymax);
        model.component("comp1").selection(tag).set("condition", "inside");
    }

    private static void material(Model model, String tag, String label, String selection,
                                 String sigma, String epsr) {
        model.component("comp1").material().create(tag, "Common");
        model.component("comp1").material(tag).label(label);
        model.component("comp1").material(tag).selection().named(selection);
        model.component("comp1").material(tag).propertyGroup("def").set("electricconductivity",
            new String[]{sigma, "0", "0", "0", sigma, "0", "0", "0", sigma});
        model.component("comp1").material(tag).propertyGroup("def").set("relpermittivity",
            new String[]{epsr, "0", "0", "0", epsr, "0", "0", "0", epsr});
    }

    private static void coupling(Model model, String tag, String type, String selection) {
        model.component("comp1").cpl().create(tag, type);
        model.component("comp1").cpl(tag).selection().named(selection);
    }

    private static void requireSelection(Model model, String tag, int dimension) {
        if (model.component("comp1").selection(tag).entities(dimension).length == 0) {
            throw new IllegalStateException("Empty selection: " + tag + ". Check geometry and selection bounds.");
        }
    }

    public static Model build() {
        Model model = ModelUtil.create("SiliconLeakage2D");
        model.label("2D-silicon leakage.mph");
        model.param().set("t_metal", "500[nm]", "Top electrode thickness; ideal conductor");
        model.param().set("t_ox", "12[um]", "Oxide thickness");
        model.param().set("t_si", "675[um]", "Silicon thickness");
        model.param().set("h_air", "750[um]", "Vacuum height above oxide surface");
        model.param().set("f_rf", "40[MHz]", "RF frequency");
        model.param().set("Vrf", "100[V]", "PEAK phasor voltage, not RMS");
        model.param().set("rho_si", "1[ohm*cm]", "Silicon bulk resistivity");
        model.param().set("eps_ox", "3.9", "Oxide relative permittivity");
        model.param().set("eps_si", "11.7", "Silicon relative permittivity");
        model.param().set("Lref", "1[mm]", "Reported out-of-plane trap length");
        model.param().set("mesh_scale", "1", "Increase above 1 for mesh refinement");

        model.component().create("comp1", true);
        model.component("comp1").geom().create("geom1", 2);
        model.component("comp1").geom("geom1").lengthUnit("um");

        List<String> metalTags = new ArrayList<String>();
        List<String> rfBoundaries = new ArrayList<String>();
        List<String> groundBoundaries = new ArrayList<String>();
        TreeSet<Double> cuts = new TreeSet<Double>();
        for (int i = 0; i < ELECTRODES.length; i++) {
            double[] e = ELECTRODES[i];
            String tag = "metal" + i;
            rectangle(model, tag, Double.toString(e[0]), "0", Double.toString(e[1]-e[0]), "t_metal");
            metalTags.add(tag);
            if (e[2] == 1) rfBoundaries.add("geom1_" + tag + "_bnd");
            else groundBoundaries.add("geom1_" + tag + "_bnd");
            cuts.add(e[0]);
            cuts.add(e[1]);
        }
        Double[] xcuts = cuts.toArray(new Double[0]);
        List<String> siDomains = new ArrayList<String>();
        List<String> oxideDomains = new ArrayList<String>();
        // Partition both layers at electrode edges. This creates exact interface
        // segments under the RF rails, without relying on hard-coded entity IDs.
        for (int i = 0; i < xcuts.length-1; i++) {
            String x = Double.toString(xcuts[i]);
            String width = Double.toString(xcuts[i+1]-xcuts[i]);
            rectangle(model, "si" + i, x, "-t_ox-t_si", width, "t_si");
            rectangle(model, "ox" + i, x, "-t_ox", width, "t_ox");
            siDomains.add("geom1_si" + i + "_dom");
            oxideDomains.add("geom1_ox" + i + "_dom");
        }
        rectangle(model, "airbox", Double.toString(xcuts[0]), "0",
                  Double.toString(xcuts[xcuts.length-1]-xcuts[0]), "h_air");
        model.component("comp1").geom("geom1").create("air", "Difference");
        model.component("comp1").geom("geom1").feature("air").selection("input").set(new String[]{"airbox"});
        model.component("comp1").geom("geom1").feature("air").selection("input2").set(strings(metalTags));
        model.component("comp1").geom("geom1").feature("air").set("keepsubtract", "on");
        model.component("comp1").geom("geom1").feature("air").set("selresult", "on");
        model.component("comp1").geom("geom1").feature("air").set("selresultshow", "all");
        model.component("comp1").geom("geom1").run(); // Form Union, keeping interior boundaries.

        union(model, "domSi", 2, strings(siDomains));
        union(model, "domOx", 2, strings(oxideDomains));
        union(model, "domFields", 2, new String[]{"domSi", "domOx", "geom1_air_dom"});
        union(model, "bndRF", 1, strings(rfBoundaries));
        union(model, "bndGround", 1, strings(groundBoundaries));
        model.component("comp1").selection("domSi").label("Silicon: finite conductivity, no backside contact");
        model.component("comp1").selection("bndRF").label("All surfaces of the two RF electrodes");

        double yInterface = -model.param().evaluate("t_ox")/1e-6;
        double yBack = -(model.param().evaluate("t_ox") + model.param().evaluate("t_si"))/1e-6;
        double tol = 1e-3; // um, only for selecting existing geometry entities.
        List<String> rfInterface = new ArrayList<String>();
        for (int i = 0; i < ELECTRODES.length; i++) {
            if (ELECTRODES[i][2] != 1) continue;
            String tag = "ifRF" + i;
            boundaryBox(model, tag, ELECTRODES[i][0]-tol, ELECTRODES[i][1]+tol,
                        yInterface-tol, yInterface+tol);
            rfInterface.add(tag);
        }
        union(model, "bndSiRF", 1, strings(rfInterface));
        boundaryBox(model, "bndInterface", xcuts[0]-tol, xcuts[xcuts.length-1]+tol,
                    yInterface-tol, yInterface+tol);
        boundaryBox(model, "bndBack", xcuts[0]-tol, xcuts[xcuts.length-1]+tol,
                    yBack-tol, yBack+tol);
        model.component("comp1").selection("bndBack").label("Floating backside: default Electric Insulation");
        for (String tag : new String[]{"domSi", "domOx", "domFields"}) requireSelection(model, tag, 2);
        for (String tag : new String[]{"bndRF", "bndGround", "bndSiRF", "bndInterface", "bndBack"}) requireSelection(model, tag, 1);

        material(model, "matSi", "Silicon, 1/rho_si", "domSi", "1/rho_si", "eps_si");
        material(model, "matOx", "Ideal SiO2", "domOx", "0[S/m]", "eps_ox");
        material(model, "matAir", "Vacuum", "geom1_air_dom", "0[S/m]", "1");

        // Frequency-domain Electric Currents includes displacement current.
        // Metal volumes are excluded; the actual 500 nm metal surfaces are equipotential.
        model.component("comp1").physics().create("ec", "ConductiveMedia", "geom1");
        model.component("comp1").physics("ec").selection().named("domFields");
        model.component("comp1").physics("ec").prop("d").set("d", "Lref");
        model.component("comp1").physics("ec").create("potRF", "ElectricPotential", 1);
        model.component("comp1").physics("ec").feature("potRF").selection().named("bndRF");
        model.component("comp1").physics("ec").feature("potRF").set("V0", "Vrf");
        model.component("comp1").physics("ec").create("gndTop", "Ground", 1);
        model.component("comp1").physics("ec").feature("gndTop").selection().named("bndGround");
        // All remaining exterior boundaries retain default Electric Insulation.
        // Do NOT put Ground or Floating Potential on the finite-resistivity Si domain.

        model.component("comp1").variable().create("lossSi");
        model.component("comp1").variable("lossSi").selection().named("domSi");
        model.component("comp1").variable("lossSi").set("qSi",
            "0.5*(1/rho_si)*(abs(d(V,x))^2+abs(d(V,y))^2)", "Time-averaged silicon Joule loss density");
        coupling(model, "intSi", "Integration", "domSi");
        coupling(model, "aveSiRF", "Average", "bndSiRF");
        coupling(model, "intSiRF", "Integration", "bndSiRF");
        coupling(model, "intInterface", "Integration", "bndInterface");

        model.component("comp1").mesh().create("mesh1");
        model.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
        model.component("comp1").mesh("mesh1").feature("size").set("hmax", "60[um]/mesh_scale");
        model.component("comp1").mesh("mesh1").feature("size").set("hmin", "0.03[um]/mesh_scale");
        model.component("comp1").mesh("mesh1").feature("size").set("hgrad", 1.2);
        model.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
        model.component("comp1").mesh("mesh1").feature("ftri1").selection().named("domFields");
        model.component("comp1").mesh("mesh1").feature("ftri1").create("sizeOx", "Size");
        model.component("comp1").mesh("mesh1").feature("ftri1").feature("sizeOx").selection().geom("geom1", 2);
        model.component("comp1").mesh("mesh1").feature("ftri1").feature("sizeOx").selection().named("domOx");
        model.component("comp1").mesh("mesh1").feature("ftri1").feature("sizeOx").set("custom", "on");
        model.component("comp1").mesh("mesh1").feature("ftri1").feature("sizeOx").set("hmaxactive", true);
        model.component("comp1").mesh("mesh1").feature("ftri1").feature("sizeOx").set("hmax", "1.5[um]/mesh_scale");
        model.component("comp1").mesh("mesh1").run();

        model.study().create("std1");
        model.study("std1").label("40 MHz: floating silicon");
        model.study("std1").create("freq", "Frequency");
        model.study("std1").feature("freq").set("plist", "f_rf");
        return model;
    }

    private static void results(Model model) {
        // Called after the first solve, when dset1 exists.
        model.result().table().create("tblLeak", "Table");
        model.result().table("tblLeak").label("RF coupling and Si dissipation");
        model.result().numerical().create("evalLeak", "EvalGlobal");
        model.result().numerical("evalLeak").set("data", "dset1");
        model.result().numerical("evalLeak").set("expr", new String[]{
            "comp1.aveSiRF(abs(comp1.V))",
            "100*comp1.aveSiRF(abs(comp1.V))/Vrf",
            "Lref*comp1.intSi(comp1.qSi)",
            "abs(Lref*comp1.intSiRF(-comp1.ec.Jy))",
            "abs(Lref*comp1.intInterface(comp1.ec.Jy))"
        });
        model.result().numerical("evalLeak").set("unit", new String[]{"V", "1", "mW", "mA", "mA"});
        model.result().numerical("evalLeak").set("descr", new String[]{
            "Mean Si RF voltage beneath rails (peak)", "RF voltage fraction (%)",
            "Si loss for Lref (time averaged)", "Local interface RF current for Lref (peak)",
            "Net interface current for Lref (should approach zero)"
        });
        model.result().numerical("evalLeak").set("table", "tblLeak");
        String[] tags = {"pgVoltage", "pgLoss"};
        String[] labels = {"Silicon RF voltage amplitude", "Silicon time-averaged Joule heating"};
        String[] expressions = {"abs(comp1.V)", "comp1.qSi"};
        String[] units = {"V", "W/m^3"};
        for (int i = 0; i < tags.length; i++) {
            model.result().create(tags[i], "PlotGroup2D");
            model.result(tags[i]).label(labels[i]);
            model.result(tags[i]).set("data", "dset1");
            model.result(tags[i]).create("surf1", "Surface");
            model.result(tags[i]).feature("surf1").set("expr", expressions[i]);
            model.result(tags[i]).feature("surf1").set("unit", units[i]);
            model.result(tags[i]).feature("surf1").selection().set(
                model.component("comp1").selection("domSi").entities(2));
        }
    }

    public static Model run() {
        ModelUtil.showProgress(true);
        Model model = build();
        File out = new File("silicon_leakage_comsol").getAbsoluteFile();
        if (!out.isDirectory() && !out.mkdirs()) throw new IllegalStateException("Cannot create " + out);
        // Separate solves make each resistivity's solution explicit and easy to inspect.
        // Each saved MPH is fully editable in COMSOL Desktop.
        try (PrintWriter csv = new PrintWriter(new File(out, "silicon_leakage_summary.csv"), "UTF-8")) {
            model.save(new File(out, "silicon_leakage_setup.mph").getPath());
            csv.println("rho_ohm_cm,reference_length_mm,si_mean_peak_V,voltage_fraction_percent,si_loss_mW,interface_current_peak_mA,net_interface_current_peak_mA");
            double referenceLengthMM = model.param().evaluate("Lref")/1e-3;
            for (int i = 0; i < RESISTIVITIES.length; i++) {
                double rho = RESISTIVITIES[i];
                model.param().set("rho_si", Double.toString(rho) + "[ohm*cm]");
                model.study("std1").run();
                if (i == 0) results(model);
                model.result().numerical("evalLeak").setResult();
                double[][] values = model.result().numerical("evalLeak").getReal();
                if (values.length != 5 || values[0].length != 1) {
                    throw new IllegalStateException("Expected five expressions at one frequency.");
                }
                csv.printf(Locale.US, "%.8g,%.8g,%.12g,%.12g,%.12g,%.12g,%.12g%n",
                    rho, referenceLengthMM, values[0][0], values[1][0], values[2][0], values[3][0], values[4][0]);
                csv.flush();
                System.out.printf(Locale.US, "rho=%.3g ohm cm: V_Si=%.6g V peak, P_Si=%.6g mW for %.6g mm%n",
                    rho, values[0][0], values[2][0], referenceLengthMM);
                String rhoName = String.format(Locale.US, "%s", rho).replace('.', 'p');
                model.save(new File(out, "silicon_leakage_rho_" + rhoName + "_ohm_cm.mph").getPath());
            }
            model.save(new File(out, "2D_silicon_leakage.mph").getPath());
        } catch (java.io.IOException e) {
            throw new RuntimeException("Could not save COMSOL results in " + out, e);
        }
        return model;
    }

    public static void main(String[] args) {
        run(); // COMSOL batch initializes the COMSOL runtime before calling this class.
    }
}
