var oldCallback;
var score = 0;
var num_trials = 20;

function sendData() {
    var submit = confirm("Would you like to submit your data and leave this site?");
    if(submit){
	console.log('sending data to mturk');
        jsPsych.turk.submitToTurk({'score':score});
    }
}

var consentHTML = {
    'str1' : '<p>In this HIT, you will view some drawings produced by children who were trying to trace a shape as accurately as they could. Your task is to rate each tracing on a 5-point scale. </p>',
    'str2' : '<p>We expect this hit to take approximately 10-15 minutes to complete, including the time it takes to read instructions.</p>',
    'str3' : "<p>If you encounter a problem or error, send us an email (sketchloop@gmail.com) and we will make sure you're compensated for your time! Please pay attention and do your best! Thank you!</p><p> Note: We recommend using Chrome. We have not tested this HIT in other browsers.</p>",
    'str4' : ["<u><p id='legal'>Consenting to Participate:</p></u>",
        "<p id='legal'>By completing this HIT, you are participating in a study being performed by cognitive scientists in the Stanford Department of Psychology. If you have questions about this research, please contact the <b>Sketchloop Admin</b> at <b><a href='mailto://sketchloop@gmail.com'>sketchloop@gmail.com</a> </b> or Zixian Chai (zchai14@stanford.edu) You must be at least 18 years old to participate. Your participation in this research is voluntary. You may decline to answer any or all of the following questions. You may decline further participation, at any time, without adverse consequences. Your anonymity is assured; the researchers who have requested your participation will not receive any personal information about you.</p>"].join(' ')
};

var instructionsHTML = {
    'str1' : "<p> Here’s how the game will work: </p> <p> On each trial, you will see a tracing on top of a reference shape. The tracing is marked in red and the reference shape is in grey. Your goal is to rate how accurately the tracing matches the SHAPE and is aligned to the POSITION of the reference. The rating scale ranges from 1 (POOR) to 5 (EXCELLENT).</p>",
    'str2': ["<p>Here’s how the game will work: </p> <p> On each trial, you will see a tracing on top of a reference shape. The tracing is marked in red and the reference shape is in grey. Your goal is to rate how accurately the tracing matches the SHAPE and is aligned to the POSITION of the reference. The rating scale ranges from 1 (POOR) to 5 (EXCELLENT). </p> <p> Here are some example tracings that should be given a score of 5 (EXCELLENT) and some tracings that should be given a score of 1 (POOR).</p>",
        '<p>Example tracing with score 5: </p>',
              '<div class="eg_div"><img class="eg_img" src="img/t5_square.png"><img class="eg_img" src="img/t5_shape.png"><img class="eg_img" src="img/t5_circle.png"></div>',
              '<p>Example tracing with score 1: </p>',
	      '<div class="eg_div"><img class="eg_img" src="img/t1_square.png"><img class="eg_img" src="img/t1_shape.png"><img class="eg_img" src="img/t1_circle.png"></div>'].join(' '),
    'str3': ['<p> If you notice any of the following, this should reduce the score you assign to that tracing:</p>',
        '<ul><li>Adding extra objects to the tracing (e.g. scribbles, heart, flower, smiling faces)<img class="notice_img" src="img/extra.png"></li>',
        '<li>Painting or "filling in" the reference shape, rather than tracing its outline<img class="notice_img" src="img/paint.png"></li></ul>',].join(' '),
    'str4':'<p> A different sketch will appear on each trial. After a brief two-second delay, the buttons will become active (dark gray) so you can submit your rating. Please take your time to provide as accurate of a rating as you can.</p> </p> <img class="rating_img" src="img/rating.png">',
    'str5': "<p> When you finish, please click the submit button to finish the game . Let's begin!"
};



var welcomeTrial = {
    type: 'instructions',
    pages: [
        consentHTML.str1,
        consentHTML.str2,
        consentHTML.str3,
        consentHTML.str4,
        instructionsHTML.str1,
        instructionsHTML.str2,
        instructionsHTML.str3,
	    instructionsHTML.str4,
        instructionsHTML.str5
    ],
    show_clickable_nav: true
};

var acceptHTML = {
    'str1' : '<p> Welcome! In this HIT, you will see some sketches of objects. For each sketch, you will try to guess which of the objects is the best match. </p>',
    'str2' : '<p> This is only a demo! If you are interested in participating, please accept the HIT in MTurk before continuing further. </p>'
}

var previewTrial = {
    type: 'instructions',
    pages: [acceptHTML.str1, acceptHTML.str2],
    show_clickable_nav: true,
    allow_keys: false
}

var goodbyeTrial = {
    type: 'instructions',
    pages: [
        '<p> Once you click the submit button, you will be prompted with a pop-up asking you if you are sure that you want to leave the site. Please click the OK button, which will trigger submission of this HIT to Amazon Mechanical Turk. </p>'
    ],
    show_clickable_nav: true,
    allow_backward:false,
    button_label_next: 'Submit the HIT',
    on_finish: function() { sendData();}
};

// define trial object with boilerplate
function Trial () {
    this.type = 'image-button-response';
    this.iterationName = 'pilot0';
    this.dev_mode = false;
    this.prompt = "Please rate how well this tracing matches the reference shape.";
    this.image_url = "/demo.png";
    this.category ='square';
    this.choices = ['1','2','3','4','5'];
    this.dev_mode = false
}

function setupGame () {

    // number of trials to fetch from database is defined in ./app.js
    var socket = io.connect();
    socket.on('onConnected', function(d) {
        // get workerId, etc. from URL (so that it can be sent to the server)
        var turkInfo = jsPsych.turk.turkInfo();

        // pull out info from server
        var id = d.id;

        // at end of each trial save score locally and send data to server
        var main_on_finish = function(data) {
            if (data.score) {
                score = data.score;
            }
            socket.emit('currentData', data);
        };

        var main_on_start = function(trial) {

            oldCallback = newCallback;
            var newCallback = function(d) {
                trial.category = d.category;
                trial.image_url = d.img_url;
                trial.age = d.age;
                trial.session_id = d.session_id;
                trial.choices = _.range(1, d.number_rating_levels+1);
                trial.upper_bound = d.upper_bound;
                trial.lower_bound = d.lower_bound;

            };
            socket.removeListener('stimulus', oldCallback);
            socket.on('stimulus', newCallback);
            // call server for stims
            socket.emit('getStim', {gameID: id});
        };

        // Bind trial data with boilerplate
        var trials = _.map(_.rangeRight(num_trials), function(trialData, i) {
            return _.extend(new Trial, trialData, {
                gameID: id,
                trialNum : i,
                post_trial_gap: 1000, // add brief ITI between trials
                on_start: main_on_start,
                on_finish : main_on_finish

            });
        });

	
        // Stick welcome trial at beginning & goodbye trial at end
        if (!turkInfo.previewMode) {
            trials.unshift(welcomeTrial);
        } else {
            trials.unshift(previewTrial); // if still in preview mode, tell them to accept first.
        }
        trials.push(goodbyeTrial);

        jsPsych.init({
            timeline: trials,
            default_iti: 1000,
            show_progress_bar: true
        });
    });


}
